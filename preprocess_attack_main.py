# Copyright 2022 Google LLC
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     https://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.

"""Main script for running attacks on ML models with preprocessors."""

from __future__ import annotations

import os
import pickle
import pprint
import random
import sys
import time
from copy import deepcopy

import attack_prep.utils.backward_compat  # pylint: disable=unused-import

# pylint: disable=wrong-import-order
import numpy as np
import torch
import torchvision
from torch import nn
from torch.backends import cudnn
from torch.nn import Identity

from attack_prep.attack import ATTACK_DICT, smart_noise
from attack_prep.attack.base import Attack
from attack_prep.attack.util import find_preimage, select_targets
from attack_prep.preprocessor.base import Preprocessor
from attack_prep.preprocessor.util import setup_preprocessor
from attack_prep.utils.argparser import parse_args
from attack_prep.utils.dataloader import get_dataloader
from attack_prep.utils.model import setup_model

_DataLoader = torch.utils.data.DataLoader

_HUGE_NUM = 1e9


def _compute_dist(
    images: torch.Tensor, x_adv: torch.Tensor, order: str
) -> torch.Tensor:
    """Compute distance between images and x_adv."""
    dist: torch.Tensor
    diff = images - x_adv
    num_samples = len(images)
    if order == "2":
        diff.square_()
        dist = diff.view(num_samples, -1).sum(1)
        dist.sqrt_()
    elif order == "inf":
        diff.abs_()
        dist = diff.view(num_samples, -1).max(1)[0]
    else:
        raise NotImplementedError(
            f'Invalid norm; p must be "2", but it is {order}.'
        )
    return dist.cpu()


def _print_result(
    name: str,
    config: dict[str, str | int | float],
    images: torch.Tensor,
    labels: torch.Tensor,
    x_adv: torch.Tensor,
    y_pred_adv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    print(f"=> Attack {name}...")
    targeted: bool = config["targeted"]
    order: str = config["ord"]
    idx_success = y_pred_adv != labels
    if targeted:
        idx_success.logical_not_()
    print(f"   success rate: {idx_success.float().mean().item():.4f}")
    dist = _compute_dist(images, x_adv, order)
    dist_success = dist[idx_success]
    print(f"   mean dist: {dist_success.mean().item():.6f}")
    # Account for small numerical error (0.01%)
    idx_success_dist = (dist <= config["epsilon"] * (1 + 1e-5)) & idx_success
    print(
        f'   success rate w/ eps={config["epsilon"]}: '
        f"{idx_success_dist.float().mean().item():.4f}"
    )
    return idx_success, dist


def _setup_smart_noise(
    lr_size: int, hr_size: int, preprocessor: nn.Module
) -> smart_noise.SmartNoise:
    """Initialize Smart Noise module from Gao et al. [2021].

    Please refer to https://github.com/wi-pi/rethinking-image-scaling-attacks
    for original implementation and detailed description.

    Args:
        lr_size: Target resizing size.
        hr_size: Original input size.
        preprocessor: Preprocessor to attack with Smart Noise.

    Returns:
        Smart Noise module to be used with HSJA or QEBA attacks.
    """
    # Load pooling layer (exact)
    pooling_layer = None
    # if args.defense != 'none':
    #     pooling_layer = POOLING_MAPS[args.defense].from_api(scaling)

    # Load pooling layer (estimate)
    pooling_layer_estimate = pooling_layer
    # if args.defense == 'median' and not args.no_smart_median:
    #     pooling_layer_estimate = POOLING_MAPS['quantile'].like(pooling_layer)

    # Load scaling layer
    # We use our preprocessor module. No need to use matrix approximation unelss
    # we move to non-pytorch resizing.
    scaling_layer = preprocessor

    # Synthesize projection (only combine non-None layers)
    projection = nn.Sequential(*filter(None, [pooling_layer, scaling_layer]))
    projection_estimate = nn.Sequential(
        *filter(None, [pooling_layer_estimate, scaling_layer])
    )

    # Smart noise
    snoise: smart_noise.SmartNoise = smart_noise.SmartNoise(
        hr_shape=(3, hr_size, hr_size),
        lr_shape=(3, lr_size, lr_size),
        projection=projection,
        projection_estimate=projection_estimate,
        precise=False,  # Don't run expensive exact projection
    )
    return snoise


def _main(config: dict[str, str | float | int], savename: str) -> None:
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])

    # Setting benchmark to True may result in non-deterministic results with
    # resizing.
    cudnn.benchmark = False
    # Setting deterministic must be set to True for neural-based preprocessor.
    # Othwerwise, the preprocessor itself may be non-deterministic.
    cudnn.deterministic = any(
        prep in config["preprocess"] for prep in ("neural", "sr", "denoise")
    )

    device: str = "cuda"
    num_samples: int = config["num_samples"]
    bypass: bool = config["bypass"]
    known_prep: bool = bypass or config["bias"]
    targeted: bool = config["targeted"]
    if bypass:
        attack_name = "Bypassing"
    elif known_prep:
        attack_name = "Biased-Gradient"
    else:
        attack_name = "Preprocessor-unaware"
    if not known_prep:
        assert not config["prep_backprop"] and not config["prep_proj"], (
            "For preprocessor-unaware attack, prep_backprop and prep_proj must "
            "be False. These two options are only compatible with Bypassing "
            "and Biased-Gradient attacks."
        )

    print("=> Setting up model and preprocessor...")
    model, preprocess = setup_model(config, device=device)
    prep, _ = preprocess.get_prep()
    model = nn.DataParallel(model).to(device).eval()
    prep = nn.DataParallel(prep).to(device).eval()
    prepare_atk_img: nn.Module = prep if bypass else Identity()

    # Used for testing attacks with our guess on the preprocessor is wrong
    mismatch_prep: str | None = config["mismatch_prep"]
    use_wrong_prep: bool = mismatch_prep is not None
    wrong_preprocess: Preprocessor | None = None
    if use_wrong_prep:
        print(f"=> Simulating mismatched preprocessing: {mismatch_prep}.")
        wrong_config = deepcopy(config)
        wrong_config["resize_out_size"] = int(mismatch_prep.split("-")[0])
        wrong_config["resize_interp"] = mismatch_prep.split("-")[1]
        wrong_preprocess: Preprocessor = setup_preprocessor(wrong_config)
        prepare_atk_img, _ = wrong_preprocess.get_prep()

    validloader: _DataLoader = get_dataloader(config)
    # Create another dataloader for targeted attacks
    targeted_dataloader: _DataLoader | None = None
    if targeted:
        print("=> Creating the second dataloader for targeted attack...")
        copy_args = deepcopy(config)
        copy_args["batch_size"] = 1
        targeted_dataloader = get_dataloader(copy_args)

    # Set up Gao et al. Smart Noise attack
    snoise: smart_noise.SmartNoise | None = None
    if config["smart_noise"]:
        print("=> Using Smart Noise...")
        snoise = _setup_smart_noise(
            preprocess.output_size, config["orig_size"], prep
        )

    # Initialize attacks with known and unknown preprocessing
    print(f"=> Initializing {attack_name} Attack...")
    attack: Attack = ATTACK_DICT[config["attack"]](
        model,
        config,
        input_size=(
            preprocess.output_size if known_prep else config["orig_size"]
        ),
        preprocess=prep if config["bias"] else None,
        prep_backprop=config["prep_backprop"],
        smart_noise=snoise,
    )

    x_gt, y_gt, y_tgt = [], [], []
    y_adv, xz_adv = [], []
    num_correct: int = 0
    num_total: int = 0
    start_time = time.time()

    # Enable grad only for white-box grad attack
    with torch.set_grad_enabled(config["attack"] == "fmn"):
        for i, (images, labels) in enumerate(validloader):
            start_batch = time.time()
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if i == 0:
                # Print out some shapes
                atk_images: torch.Tensor = prepare_atk_img(images)
                print(f"=> Labels shape: {tuple(labels.shape)}.")
                print(f"=> Orig image shape: {tuple(images.shape)}.")
                print(f"=> Attack image shape: {tuple(atk_images.shape)}.")

            # Select images that are correctly classified only (mainly to deal
            # with square attack).
            y_orig = model(images).argmax(-1)  # pylint: disable=not-callable
            idx = y_orig == labels
            num_correct += idx.int().sum()
            num_total += images.shape[0]
            if not idx.any():
                continue
            images, labels = images[idx], labels[idx]
            x_gt.append(images.cpu())
            y_gt.append(labels.cpu())
            atk_images: torch.Tensor = prepare_atk_img(images)
            atk_images.clamp_(0, 1)

            tgt_data: tuple[torch.Tensor, torch.Tensor] | None = None
            if targeted:
                # Randomly select target samples for targeted attack
                tgt_data = select_targets(model, targeted_dataloader, labels)
                tgt_data = (prepare_atk_img(tgt_data[0]), tgt_data[1])
                y_tgt.append(tgt_data[1].cpu())

            preprocess.set_x_orig(images)
            out = attack.run(atk_images, labels, tgt=tgt_data)
            xz_adv.append(out.cpu())
            # pylint: disable=not-callable
            y_adv.append(model(out.to(device)).argmax(1).cpu())

            print(
                f"batch {i + 1:4d} ({num_correct:4d}/{num_samples:4d}) | "
                f"time: {time.time() - start_batch:.2f}s",
                flush=True,
            )
            if num_correct >= num_samples:
                break

    x_gt = torch.cat(x_gt, dim=0)[:num_samples]
    y_gt = torch.cat(y_gt, dim=0)[:num_samples]
    xz_adv = torch.cat(xz_adv, dim=0)[:num_samples]
    y_adv = torch.cat(y_adv, dim=0)[:num_samples]
    if targeted:
        y_tgt = torch.cat(y_tgt, dim=0)[:num_samples]

    # xz_adv holds output from attack. It's in original space for unknown-prep
    # attack, but it could be in either original or processed space for known-
    # prep attack for Biased-Gradient and Bypassing, respectively.
    x_adv = xz_adv
    y_proj: torch.Tensor | None = None  # Prediction after recovery (projection)
    if known_prep:
        x_adv, y_proj = x_gt.clone(), y_gt.clone()
        # Briefly put prep on cpu since xz_adv is on cpu
        cpu_prep = prep.module.to(xz_adv.device)
        z_adv = cpu_prep(xz_adv)
        prep.module.to(device)
        # Find pre-image projection of known-preprocessing attack
        batch_size = 1
        num_batches = int(np.ceil(num_samples / batch_size))
        for i in range(num_batches):
            begin, end = i * batch_size, (i + 1) * batch_size
            y_atk = y_tgt[begin:end] if targeted else y_gt[begin:end]
            out = find_preimage(
                config,
                model,
                y_atk.to(device),
                x_gt[begin:end].to(device),
                z_adv[begin:end].to(device),
                wrong_preprocess if use_wrong_prep else preprocess,
                verbose=config["verbose"],
            )
            x_adv[begin:end] = out.cpu()
            with torch.no_grad():
                # pylint: disable=not-callable
                y_proj[begin:end] = model(out.to(device)).argmax(1).cpu()

    print(f"=> Total attack time: {time.time() - start_time:.2f}s")
    print(f"=> Original acc: {num_correct / num_total:.4f}")
    output_dict = {"args": config}

    idx_success, dist = _print_result(
        f"{attack_name} Attack",
        config,
        x_gt,
        y_tgt if targeted else y_gt,
        x_adv,
        y_proj if known_prep else y_adv,
    )

    if known_prep and not bypass:
        # Print results before recovery phase if possible (no dim change)
        idx_success_nr, dist_nr = _print_result(
            f"{attack_name} Attack (no recovery)",
            config,
            x_gt,
            y_tgt if targeted else y_gt,
            xz_adv,
            y_adv,
        )

        # Select smaller distance between with and without recovery
        dist_nr[~idx_success_nr] += _HUGE_NUM
        dist[~idx_success] += _HUGE_NUM
        idx_success = idx_success | idx_success_nr
        dist = torch.minimum(dist_nr, dist)

    output_dict["idx_success"] = idx_success
    output_dict["dist"] = dist

    if config["save_adv"]:
        num_save_img: int = 32
        output_dict["x_gt"] = x_gt
        output_dict["y_gt"] = y_gt
        output_dict["y_tgt"] = y_tgt
        output_dict["x_adv"] = x_adv
        output_dict["z_adv"] = z_adv
        torchvision.utils.save_image(x_gt[:num_save_img], "x_gt.png")
        torchvision.utils.save_image(x_adv[:num_save_img], "x_adv_ukp.png")
        torchvision.utils.save_image(z_adv[:num_save_img], "z_adv_kp.png")

    with open(savename + ".pkl", "wb") as file:
        pickle.dump(output_dict, file)
    print("Finished.")


def run_one_setting(config: dict[str, str | int | float]) -> None:
    """Run attack for one setting given by config."""
    # Determine output file name
    # Get all preprocessings and their params
    preps = config["preprocess"].split("-")
    prep_params = ""
    for key in sorted(config.keys()):
        key_prep_name = key.split("_")[0]
        # Skip this key
        if key in ("sr_config_path", "denoise_config_path"):
            continue
        for prep in preps:
            if prep == key_prep_name:
                prep_params += f"-{config[key]}"

    atk_params = ""
    for key in sorted(config.keys()):
        if config["attack"] == key.split("_", maxsplit=1)[0]:
            atk_params += f"-{config[key]}"

    tokens = [
        f'{config["preprocess"]}{prep_params}',
        f'orig{config["orig_size"]}',
        f'eps{config["epsilon"]}',
    ]
    if config["targeted"]:
        tokens.append("tg")
    if config["name"]:
        tokens.append(config["name"])
    tokens.append(f'{config["attack"]}{atk_params}')
    if config["mismatch_prep"] is not None:
        tokens.append(f'mm-{config["mismatch_prep"]}')
    if config["smart_noise"]:
        tokens.append("sns")
    if config["bypass"]:
        tokens.append("bypass")  # Bypassing Attack
    if config["bias"]:
        tokens.append("bias")  # Biased-Gradient Attac
    if config["prep_backprop"]:
        tokens.append("bp")
    path = f'./results/{"-".join(tokens)}'

    # Redirect output if not debug
    if not config["debug"]:
        print(f"Output is being written to {path}.out", flush=True)
        sys.stdout = open(path + ".out", "w", encoding="utf-8")
        sys.stderr = sys.stdout
    print(path)

    pprint.pprint(config)
    _main(config, path)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs("./results", exist_ok=True)
    if args.debug:
        args.verbose = True
    run_one_setting(vars(args))
