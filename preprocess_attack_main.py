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
from typing import Any

import numpy as np
import timm
import torch
import torchvision
from torch import nn
from torch.backends import cudnn

from attack_prep.attack import ATTACK_DICT, smart_noise
from attack_prep.attack.util import find_preimage, select_targets
from attack_prep.preprocessor.base import Preprocessor
from attack_prep.preprocessor.util import setup_preprocessor
from attack_prep.utils.argparser import parse_args
from attack_prep.utils.dataloader import get_dataloader
from attack_prep.utils.model import PreprocessModel

_DataLoader = torch.utils.data.DataLoader

_HUGE_NUM = 1e9


def _compute_dist(
    images: torch.Tensor, x_adv: torch.Tensor, order: str
) -> torch.Tensor:
    """Compute distance between images and x_adv."""
    dist: torch.Tensor
    if order == "2":
        dist = (torch.sum((images - x_adv) ** 2, (1, 2, 3)) ** 0.5).cpu()
    elif order == "inf":
        dist = (
            (images - x_adv).abs().reshape(images.size(0), -1).max(1)[0].cpu()
        )
    else:
        raise NotImplementedError(
            f'Invalid norm; p must be "2", but it is {order}.'
        )
    return dist


def _print_result(
    name: str,
    config: dict[str, Any],
    images: torch.Tensor,
    labels: torch.Tensor,
    x_adv: torch.Tensor,
    y_pred_adv: torch.Tensor,
    order: str = "2",
) -> tuple[torch.Tensor, torch.Tensor]:
    print(f"=> Attack {name}...")
    idx_success = y_pred_adv != labels
    if config["targeted"]:
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


def _setup_smart_noise(lr_size, hr_size, preprocessor):
    # Load scaling
    # lr_size, hr_size = base, base * args.scale
    # lr_shape, hr_shape = (3, lr_size, lr_size), (3, hr_size, hr_size)
    # scaling = ScalingAPI(hr_shape[1:], lr_shape[1:], args.lib, args.alg)

    # Load pooling layer (exact)
    pooling_layer = None
    # if args.defense != 'none':
    #     pooling_layer = POOLING_MAPS[args.defense].from_api(scaling)

    # Load pooling layer (estimate)
    pooling_layer_estimate = pooling_layer
    # if args.defense == 'median' and not args.no_smart_median:
    #     pooling_layer_estimate = POOLING_MAPS['quantile'].like(pooling_layer)

    # Load scaling layer
    # scaling_layer = None
    # if args.scale > 1:
    #     scaling_layer = ScalingLayer.from_api(scaling).eval().cuda()
    # Scaling layer scales noise down and up
    prep, _, _, _ = preprocessor.get_prep()
    scaling_layer = prep

    # Synthesize projection (only combine non-None layers)
    projection = nn.Sequential(*filter(None, [pooling_layer, scaling_layer]))
    projection_estimate = nn.Sequential(
        *filter(None, [pooling_layer_estimate, scaling_layer])
    )

    # Smart noise
    snoise = smart_noise.SmartNoise(
        hr_shape=(3, hr_size, hr_size),
        lr_shape=(3, lr_size, lr_size),
        projection=projection,
        projection_estimate=projection_estimate,
        precise=False,
        # precise=args.precise_noise,
    )
    return snoise


def _main(config: dict[str, str | float | int], savename: str) -> None:

    device: str = "cuda"
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    cudnn.benchmark = True
    num_samples: int = config["num_samples"]

    print("=> Setting up base model...")
    normalize: dict[str, torch.Tensor] = {
        "mean": torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32),
        "std": torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32),
    }
    base_model: nn.Module = timm.create_model("resnet18", pretrained=True)
    preprocess: Preprocessor = setup_preprocessor(config)
    prep, _, atk_prep, prepare_atk_img = preprocess.get_prep()

    # Initialize models with known and unknown preprocessing
    ukp_model: PreprocessModel = (
        PreprocessModel(base_model, preprocess=prep, normalize=normalize)
        .eval()
        .to(device)
    )
    kp_model: PreprocessModel = (
        PreprocessModel(base_model, preprocess=atk_prep, normalize=normalize)
        .eval()
        .to(device)
    )
    ukp_model: nn.Module = nn.DataParallel(ukp_model).eval().to(device)
    kp_model: nn.Module = nn.DataParallel(kp_model).eval().to(device)

    # Used for testing attacks with our guess on the preprocessor is wrong
    use_wrong_prep: int = config["mismatch_prep"] is not None
    wrong_preprocess: Preprocessor | None = None
    if use_wrong_prep:
        print(
            f"=> Simulating mismatched preprocessing: "
            f'{config["mismatch_prep"]}'
        )
        wrong_config = deepcopy(config)
        wrong_config["resize_out_size"] = int(
            config["mismatch_prep"].split("-")[0]
        )
        wrong_config["resize_interp"] = config["mismatch_prep"].split("-")[1]
        wrong_preprocess: Preprocessor = setup_preprocessor(wrong_config)
        _, _, _, prepare_atk_img = wrong_preprocess.get_prep()

    validloader: _DataLoader = get_dataloader(config)
    # Create another dataloader for targeted attacks
    targeted_dataloader: _DataLoader | None = None
    if config["targeted"]:
        print("=> Creating the second dataloader for targeted attack...")
        copy_args = deepcopy(config)
        copy_args["batch_size"] = 1
        targeted_dataloader = get_dataloader(copy_args)

    # Set up Gao et al. Smart Noise attack
    snoise: smart_noise.SmartNoise | None = None
    if config["smart_noise"]:
        snoise = _setup_smart_noise(
            config["resize_out_size"], config["orig_size"], preprocess
        )

    # Initialize attacks with known and unknown preprocessing
    attack_init = ATTACK_DICT[config["attack"]]
    ukp_attack = attack_init(
        ukp_model,
        config,
        input_size=config["orig_size"],
        targeted_dataloader=targeted_dataloader,
        smart_noise=snoise,
    )
    if config["prep_backprop"]:
        # Only makes sense to backprop through preprocessing if it's used in
        # the forward pass.
        config["prep_grad_est"] = True
    kp_attack = attack_init(
        kp_model,
        config,
        input_size=preprocess.output_size,
        targeted_dataloader=targeted_dataloader,
        preprocess=(
            atk_prep if config["prep_grad_est"] or config["prep_proj"] else None
        ),
        prep_backprop=config["prep_backprop"],
        prep_proj=config["prep_proj"],
        smart_noise=snoise,
    )

    x_gt, y_gt, y_tgt = [], [], []
    y_pred_ukp, y_pred_kp, y_pred_proj = [], [], []
    x_adv_kp, x_adv_ukp, z_adv_kp = [], [], []
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
                print(
                    f"labels shape: {labels.shape}, orig images shape: "
                    f"{images.shape}, attack images shape: {atk_images.shape}."
                )

            # Select images that are correctly classified only (mainly to deal
            # with square attack)
            y_ = ukp_model(images).argmax(-1)  # pylint: disable=not-callable
            idx = y_ == labels
            num_correct += idx.float().sum()
            num_total += images.shape[0]
            if not idx.any():
                continue
            images = images[idx]
            labels = labels[idx]
            x_gt.append(images.cpu())
            y_gt.append(labels.cpu())
            atk_images: torch.Tensor = prepare_atk_img(images)

            tgt_data: torch.Tensor | None = None
            if config["targeted"]:
                # Randomly select target samples for targeted attack
                tgt_data = select_targets(
                    ukp_model, targeted_dataloader, labels
                )
                y_tgt.append(tgt_data[1].cpu())

            if not config["run_kp_only"]:
                # Attack preprocess model (unknown)
                out = ukp_attack.run(images, labels, tgt=tgt_data)
                x_adv_ukp.append(out.cpu())
                y_pred_ukp.append(
                    ukp_model(out.to(device))  # pylint: disable=not-callable
                    .argmax(1)
                    .cpu()
                )

            if config["targeted"]:
                tgt_data = (prepare_atk_img(tgt_data[0]), tgt_data[1])

            if not config["run_ukp_only"]:
                # Attack preprocessed input directly (known)
                preprocess.set_x_orig(images)
                atk_images.clamp_(0, 1)
                out = kp_attack.run(
                    atk_images, labels, tgt=tgt_data, preprocess=prepare_atk_img
                )
                z_adv_kp.append(out.cpu())
                y_pred_kp.append(
                    kp_model(out.to(device))  # pylint: disable=not-callable
                    .argmax(1)
                    .cpu()
                )

            print(
                f"batch {i + 1}: {time.time() - start_batch:.2f}s", flush=True
            )
            if num_correct >= num_samples:
                break

    x_gt = torch.cat(x_gt, dim=0)[:num_samples]
    y_gt = torch.cat(y_gt, dim=0)[:num_samples]
    if config["targeted"]:
        y_tgt = torch.cat(y_tgt, dim=0)[:num_samples]

    if not config["run_ukp_only"]:
        x_adv_kp = x_gt.clone()
        y_pred_proj = y_gt.clone()
        z_adv_kp = torch.cat(z_adv_kp, dim=0)[:num_samples]
        # Briefly put prep on cpu since z_adv_kp is on cpu
        prep.to(z_adv_kp.device)
        z_adv_kp = prep(z_adv_kp)
        prep.to(device)
        # Find pre-image projection of known-preprocessing attack
        batch_size = 1
        num_batches = int(np.ceil(num_samples / batch_size))
        for b in range(num_batches):
            begin, end = b * batch_size, (b + 1) * batch_size
            y = y_tgt[begin:end] if config["targeted"] else y_gt[begin:end]
            out = find_preimage(
                config,
                ukp_model,
                kp_model,
                y.to(device),
                x_gt[begin:end].to(device),
                z_adv_kp[begin:end].to(device),
                wrong_preprocess if use_wrong_prep else preprocess,
                verbose=config["verbose"],
            )
            x_adv_kp[begin:end] = out.cpu()
            with torch.no_grad():
                y_pred_proj[begin:end] = (
                    ukp_model(out.to(device))  # pylint: disable=not-callable
                    .argmax(1)
                    .cpu()
                )

    print(f"=> Total attack time: {time.time() - start_time:.2f}s")
    print(f"=> Original acc: {num_correct / num_total:.4f}")
    output_dict = {"args": config}

    if not config["run_kp_only"]:
        x_adv_ukp = torch.cat(x_adv_ukp, dim=0)[:num_samples]
        y_pred_ukp = torch.cat(y_pred_ukp, dim=0)[:num_samples]
        ukp_idx, ukp_dist = _print_result(
            "Unknown Preprocess",
            config,
            x_gt,
            y_tgt if config["targeted"] else y_gt,
            x_adv_ukp,
            y_pred_ukp,
            order=config["ord"],
        )
        output_dict["idx_success_ukp"] = ukp_idx
        output_dict["dist_ukp"] = ukp_dist

    if not config["run_ukp_only"]:
        y_pred_kp = torch.cat(y_pred_kp, dim=0)[:num_samples]
        success_idx = (
            y_pred_kp == y_tgt if config["targeted"] else y_pred_kp != y_gt
        )
        print(
            "=> Initial success rate before projection: "
            f"{success_idx.float().mean().item():.4f}"
        )
        kp_idx, kp_dist = _print_result(
            "Known Preprocess",
            config,
            x_gt,
            y_tgt if config["targeted"] else y_gt,
            x_adv_kp,
            y_pred_proj,
            order=config["ord"],
        )
        output_dict["idx_success_kp"] = kp_idx
        output_dict["dist_kp"] = kp_dist

        # Also print results before recovery phase
        if not any([p in config["preprocess"] for p in ("resize", "crop")]):
            kpnr_idx, kpnr_dist = _print_result(
                "Known Preprocess (no recovery)",
                config,
                x_gt,
                y_tgt if config["targeted"] else y_gt,
                z_adv_kp,
                y_pred_kp,
                order=config["ord"],
            )

            # Select smaller distance between with and without recovery
            kpnr_dist[~kpnr_idx] += _HUGE_NUM
            kp_dist[~kp_idx] += _HUGE_NUM
            kp_idx = kp_idx | kpnr_idx
            kp_dist = torch.minimum(kpnr_dist, kp_dist)
            output_dict["idx_success_kp"] = kp_idx
            output_dict["dist_kp"] = kp_dist

    if config["save_adv"]:
        output_dict["x_gt"] = x_gt
        output_dict["y_gt"] = y_gt
        output_dict["y_tgt"] = y_tgt
        output_dict["x_adv_ukp"] = x_adv_ukp
        output_dict["x_adv_kp"] = x_adv_kp
        torchvision.utils.save_image(x_gt[:32], "x_gt.png")
        torchvision.utils.save_image(x_adv_ukp[:32], "x_adv_ukp.png")
        torchvision.utils.save_image(z_adv_kp[:32], "z_adv_kp.png")

    with open(savename + ".pkl", "wb") as file:
        pickle.dump(output_dict, file)
    print("Finished.")


def run_one_setting(config):
    """Run attack for one setting given by args."""
    # Determine output file name
    # Get all preprocessings and their params
    preps = config["preprocess"].split("-")
    prep_params = ""
    for key in sorted(config.keys()):
        key_prep_name = key.split("_")[0]
        for prep in preps:
            if prep == key_prep_name:
                prep_params += f"-{config[key]}"
    atk_params = ""
    for key in sorted(config.keys()):
        if config["attack"] == key.split("_")[0]:
            atk_params += f"-{config[key]}"
    path = (
        f'./results/{config["preprocess"]}{prep_params}-orig{config["orig_size"]}'
        f'-eps{config["epsilon"]}-{config["attack"]}{atk_params}'
    )
    if config["targeted"]:
        path += "-tg"
    if config["mismatch_prep"] is not None:
        path += f'-mm-{config["mismatch_prep"]}'
    if config["run_ukp_only"]:
        path += "-ukp"
    if config["run_kp_only"]:
        path += "-kp"
    if config["smart_noise"]:
        path += "-sns"
    if config["prep_grad_est"]:
        path += "-bg"  # Biased Gradient
    if config["prep_backprop"]:
        path += "-bp"
    if config["name"]:
        path += f'-{config["name"]}'

    # Redirect output if not debug
    if not config["debug"]:
        print(f"Output is being written to {path}.out", flush=True)
        sys.stdout = open(path + ".out", "w", encoding="utf-8")
        sys.stderr = sys.stdout

    pprint.pprint(config)
    _main(config, path)


if __name__ == "__main__":

    args = parse_args()
    os.makedirs("./results", exist_ok=True)

    if args.debug:
        args.verbose = True

    run_one_setting(vars(args))
