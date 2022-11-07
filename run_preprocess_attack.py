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

import argparse
import os
import pickle
import random
import sys
import time
from copy import deepcopy

import numpy as np
import timm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision

from extract_prep.attack import ATTACK_DICT, smart_noise
from extract_prep.attack.util import find_preimage, select_targets
from extract_prep.preprocessor import PREPROCESSORS, Sequential
from extract_prep.preprocessor.base import Preprocessor
from extract_prep.utils.dataloader import get_dataloader
from extract_prep.utils.model import PreprocessModel

_DataLoader = torch.utils.data.DataLoader

_HUGE_NUM = 1e9


def _compute_dist(
    images: torch.Tensor, x_adv: torch.Tensor, p: str
) -> torch.Tensor:
    """Compute distance between images and x_adv."""
    dist: torch.Tensor
    if p == "2":
        dist = (torch.sum((images - x_adv) ** 2, (1, 2, 3)) ** 0.5).cpu()
    elif p == "inf":
        dist = (
            (images - x_adv).abs().reshape(images.size(0), -1).max(1)[0].cpu()
        )
    else:
        raise NotImplementedError(
            f'Invalid norm; p must be "2", but it is {p}.'
        )
    return dist


def _print_result(name, args, images, labels, x_adv, y_pred_adv, ord="2"):
    print(f"=> Attack {name}...")
    idx_success = y_pred_adv != labels
    if args["targeted"]:
        idx_success.logical_not_()
    print(f"   success rate: {idx_success.float().mean().item():.4f}")
    dist = _compute_dist(images, x_adv, ord)
    dist_success = dist[idx_success]
    print(f"   mean dist: {dist_success.mean().item():.6f}")
    # Account for small numerical error (0.01%)
    idx_success_dist = (dist <= args["epsilon"] * (1 + 1e-5)) & idx_success
    print(
        f'   success rate w/ eps={args["epsilon"]}: '
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


def _run_proj_only(
    args,
    num_samples,
    ukp_model,
    kp_model,
    preprocess,
    device="cuda",
    wrong_prep=None,
):
    prep = preprocess.get_prep()[0]
    out = pickle.load(open(args["run_proj_only"], "rb"))
    z_adv_kp = out["z_adv_kp"]
    x_gt = out["x_gt"]
    y_gt = out["y_gt"]
    y_tgt = out["y_tgt"]
    x_adv_kp = x_gt.clone()
    y_pred_proj = y_gt.clone()
    # Find pre-image projection of known-preprocessing attack
    batch_size = 1
    num_batches = int(np.ceil(num_samples / batch_size))
    for b in range(num_batches):
        begin, end = b * batch_size, (b + 1) * batch_size
        if args["targeted"]:
            y = y_tgt[begin:end]
        else:
            y = y_gt[begin:end]
        out, _ = find_preimage(
            args,
            ukp_model,
            kp_model,
            y.to(device),
            x_gt[begin:end].to(device),
            z_adv_kp[begin:end].to(device),
            wrong_prep if wrong_prep is not None else prep,
            verbose=args["verbose"],
        )
        x_adv_kp[begin:end] = out.cpu()
        with torch.no_grad():
            y_pred_proj[begin:end] = ukp_model(out.to(device)).argmax(1).cpu()

    _print_result(
        "Known Preprocess",
        args,
        x_gt,
        y_tgt if args["targeted"] else y_gt,
        x_adv_kp,
        y_pred_proj,
        ord=args["ord"],
    )


def _main(args: dict[str, str | float | int], savename: str) -> None:

    device: str = "cuda"
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    cudnn.benchmark = True
    num_samples: int = args["num_samples"]

    print("=> Setting up base model...")
    normalize: dict[str, torch.Tensor] = {
        "mean": torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32),
        "std": torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32),
    }
    base_model: nn.Module = timm.create_model("resnet18", pretrained=True)

    # Set up preprocessing
    if "-" in args["preprocess"]:
        prep_init = Sequential
    else:
        prep_init = PREPROCESSORS[args["preprocess"]]
    preprocess: Preprocessor = prep_init(args, input_size=args["orig_size"])
    prep, _, atk_prep, prepare_atk_img = preprocess.get_prep()

    use_wrong_prep = args["mismatch_prep"] is not None
    wrong_preprocess: Preprocessor | None = None
    if use_wrong_prep:
        print(
            f"=> Simulating mismatched preprocessing: "
            f'{args["mismatch_prep"]}'
        )
        wrong_args = deepcopy(args)
        wrong_args["resize_out_size"] = int(args["mismatch_prep"].split("-")[0])
        wrong_args["resize_interp"] = args["mismatch_prep"].split("-")[1]
        wrong_preprocess = prep_init(wrong_args, input_size=args["orig_size"])
        wrong_prep, _, _, prepare_atk_img = wrong_preprocess.get_prep()

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

    validloader: _DataLoader = get_dataloader(args)
    # Create another dataloader for targeted attacks
    targeted_dataloader: _DataLoader | None = None
    if args["targeted"]:
        print("=> Creating the second dataloader for targeted attack...")
        copy_args = deepcopy(args)
        copy_args["batch_size"] = 1
        targeted_dataloader = get_dataloader(copy_args)

    snoise: smart_noise.SmartNoise | None = None
    if args["smart_noise"]:
        snoise = _setup_smart_noise(
            args["resize_out_size"], args["orig_size"], preprocess
        )

    # Initialize attacks with known and unknown preprocessing
    attack_init = ATTACK_DICT[args["attack"]]
    ukp_attack = attack_init(
        ukp_model,
        args,
        input_size=args["orig_size"],
        targeted_dataloader=targeted_dataloader,
        smart_noise=snoise,
    )
    if args["prep_backprop"]:
        # Only makes sense to backprop through preprocessing if it's used in
        # the forward pass.
        args["prep_grad_est"] = True
    kp_attack = attack_init(
        kp_model,
        args,
        input_size=preprocess.output_size,
        targeted_dataloader=targeted_dataloader,
        preprocess=(
            atk_prep if args["prep_grad_est"] or args["prep_proj"] else None
        ),
        prep_backprop=args["prep_backprop"],
        prep_proj=args["prep_proj"],
        smart_noise=snoise,
    )

    x_gt, y_gt, y_tgt = [], [], []
    y_pred_ukp, y_pred_kp, y_pred_proj = [], [], []
    x_adv_kp, x_adv_ukp, z_adv_kp = [], [], []
    num_correct: int = 0
    start_time = time.time()

    if args["run_proj_only"]:
        _run_proj_only(
            args,
            num_samples,
            ukp_model,
            kp_model,
            preprocess,
            device=device,
            wrong_prep=wrong_prep if use_wrong_prep else None,
        )
        return

    # Enable grad only for white-box grad attack
    with torch.set_grad_enabled(args["attack"] == "fmn"):
        for i, (images, labels) in enumerate(validloader):
            start_batch = time.time()
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Select images that are correctly classified only (mainly to deal
            # with square attack)
            y_ = ukp_model(images).argmax(-1)  # pylint: disable=not-callable
            idx = y_ == labels
            num_correct += idx.float().sum()
            if not idx.any():
                continue
            images = images[idx]
            labels = labels[idx]
            x_gt.append(images.cpu())
            y_gt.append(labels.cpu())
            atk_images = prepare_atk_img(images)
            if i == 0:
                print(
                    f"labels shape: {labels.shape}, orig images shape: "
                    f"{images.shape}, attack images shape: {atk_images.shape}."
                )

            tgt_data: torch.Tensor | None = None
            if args["targeted"]:
                # Randomly select target samples for targeted attack
                tgt_data = select_targets(
                    ukp_model, targeted_dataloader, labels
                )
                y_tgt.append(tgt_data[1].cpu())

            if not args["run_kp_only"]:
                # Attack preprocess model (unknown)
                out = ukp_attack.run(images, labels, tgt=tgt_data)
                x_adv_ukp.append(out.cpu())
                y_pred_ukp.append(
                    ukp_model(out.to(device))  # pylint: disable=not-callable
                    .argmax(1)
                    .cpu()
                )

            if args["targeted"]:
                tgt_data = (prepare_atk_img(tgt_data[0]), tgt_data[1])

            if not args["run_ukp_only"]:
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
    if args["targeted"]:
        y_tgt = torch.cat(y_tgt, dim=0)[:num_samples]

    if not args["run_ukp_only"]:
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
            if args["targeted"]:
                y = y_tgt[begin:end]
            else:
                y = y_gt[begin:end]
            out = find_preimage(
                args,
                ukp_model,
                kp_model,
                y.to(device),
                x_gt[begin:end].to(device),
                z_adv_kp[begin:end].to(device),
                wrong_preprocess if use_wrong_prep else preprocess,
                verbose=args["verbose"],
            )
            x_adv_kp[begin:end] = out.cpu()
            with torch.no_grad():
                y_pred_proj[begin:end] = (
                    ukp_model(out.to(device))  # pylint: disable=not-callable
                    .argmax(1)
                    .cpu()
                )

    print(f"=> Total attack time: {time.time() - start_time:.2f}s")
    print(f'=> Original acc: {num_correct / (i + 1) / args["batch_size"]:.4f}')
    output_dict = {"args": args}

    if not args["run_kp_only"]:
        x_adv_ukp = torch.cat(x_adv_ukp, dim=0)[:num_samples]
        y_pred_ukp = torch.cat(y_pred_ukp, dim=0)[:num_samples]
        ukp_idx, ukp_dist = _print_result(
            "Unknown Preprocess",
            args,
            x_gt,
            y_tgt if args["targeted"] else y_gt,
            x_adv_ukp,
            y_pred_ukp,
            ord=args["ord"],
        )
        output_dict["idx_success_ukp"] = ukp_idx
        output_dict["dist_ukp"] = ukp_dist

    if not args["run_ukp_only"]:
        y_pred_kp = torch.cat(y_pred_kp, dim=0)[:num_samples]
        success_idx = (
            y_pred_kp == y_tgt if args["targeted"] else y_pred_kp != y_gt
        )
        print(
            "=> Initial success rate before projection: "
            f"{success_idx.float().mean().item():.4f}"
        )
        kp_idx, kp_dist = _print_result(
            "Known Preprocess",
            args,
            x_gt,
            y_tgt if args["targeted"] else y_gt,
            x_adv_kp,
            y_pred_proj,
            ord=args["ord"],
        )
        output_dict["idx_success_kp"] = kp_idx
        output_dict["dist_kp"] = kp_dist

        # Also print results before recovery phase
        if not any([p in args["preprocess"] for p in ("resize", "crop")]):
            kpnr_idx, kpnr_dist = _print_result(
                "Known Preprocess (no recovery)",
                args,
                x_gt,
                y_tgt if args["targeted"] else y_gt,
                z_adv_kp,
                y_pred_kp,
                ord=args["ord"],
            )

            # Select smaller distance between with and without recovery
            kpnr_dist[~kpnr_idx] += _HUGE_NUM
            kp_dist[~kp_idx] += _HUGE_NUM
            kp_idx = kp_idx | kpnr_idx
            kp_dist = torch.minimum(kpnr_dist, kp_dist)
            output_dict["idx_success_kp"] = kp_idx
            output_dict["dist_kp"] = kp_dist

    if args["save_adv"]:
        output_dict["x_gt"] = x_gt
        output_dict["y_gt"] = y_gt
        output_dict["y_tgt"] = y_tgt
        output_dict["x_adv_ukp"] = x_adv_ukp
        output_dict["x_adv_kp"] = x_adv_kp
        torchvision.utils.save_image(x_gt[:32], "x_gt.png")
        torchvision.utils.save_image(x_adv_ukp[:32], "x_adv_ukp.png")
        torchvision.utils.save_image(z_adv_kp[:32], "z_adv_kp.png")

    pickle.dump(output_dict, open(savename + ".pkl", "wb"))
    print("Finished.")


def run_one_setting(args):
    """Run attack for one setting given by args."""
    # Determine output file name
    # Get all preprocessings and their params
    preps = args["preprocess"].split("-")
    prep_params = ""
    for key in sorted(args.keys()):
        key_prep_name = key.split("_")[0]
        for prep in preps:
            if prep == key_prep_name:
                prep_params += f"-{args[key]}"
    atk_params = ""
    for key in sorted(args.keys()):
        if args["attack"] == key.split("_")[0]:
            atk_params += f"-{args[key]}"
    path = (
        f'./results/{args["preprocess"]}{prep_params}-orig{args["orig_size"]}'
        f'-eps{args["epsilon"]}-{args["attack"]}{atk_params}'
    )
    if args["targeted"]:
        path += "-tg"
    if args["mismatch_prep"] is not None:
        path += f'-mm-{args["mismatch_prep"]}'
    if args["run_ukp_only"]:
        path += "-ukp"
    if args["run_kp_only"]:
        path += "-kp"
    if args["smart_noise"]:
        path += "-sns"
    if args["prep_grad_est"]:
        path += "-bg"  # Biased Gradient
    if args["prep_backprop"]:
        path += "-bp"
    if args["name"]:
        path += f'-{args["name"]}'

    # Redirect output if not debug
    if not args["debug"]:
        print(f"Output is being written to {path}.out", flush=True)
        sys.stdout = open(path + ".out", "w")
        sys.stderr = sys.stdout

    print(args)
    _main(args, path)


def parse_args() -> dict[str, str | float | int]:
    """Get a common argparser."""
    parser = argparse.ArgumentParser(
        description="Known-preprocessor Attack", add_help=True
    )
    parser.add_argument(
        "--name",
        default="",
        type=str,
        help="Custom experiment name for saving results.",
    )
    parser.add_argument(
        "--preprocess", required=True, type=str, help="Specify preprocessor."
    )
    parser.add_argument(
        "--attack",
        default="hopskipjump",
        type=str,
        help="Base attack algorithm to use.",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num-classes", default=1000, type=int)
    parser.add_argument(
        "--num-samples",
        default=16,
        type=int,
        help="Number of test samples to run attack on.",
    )
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument(
        "--workers",
        default=8,
        type=int,
        help="Number of workers for loading data.",
    )
    parser.add_argument("--data-dir", default="./data/", type=str)
    parser.add_argument(
        "--ord",
        default="2",
        type=str,
        help='Lp-norm of attack. Only L2 ("2") is supported at the moment.',
    )
    parser.add_argument("--epsilon", default=8 / 255, type=float)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--orig-size", default=224, type=int, help="Original image size."
    )
    parser.add_argument("--binary-search-steps", default=10, type=int)
    parser.add_argument("--lambda-search-steps", default=10, type=int)
    parser.add_argument(
        "--run-ukp-only",
        action="store_true",
        help="Run unknown-preprocessor attack only.",
    )
    parser.add_argument(
        "--run-kp-only",
        action="store_true",
        help="Run known-preprocessor attack only.",
    )
    parser.add_argument("--run-proj-only", default="", type=str)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save-adv", action="store_true")
    parser.add_argument("--mismatch-prep", default=None, type=str)
    parser.add_argument("--targeted", action="store_true")
    parser.add_argument(
        "--prep-grad-est",
        action="store_true",
        help=(
            "Apply preprocessing during attack gradient estimation (only "
            "works with HSJ, QEBA)."
        ),
    )
    parser.add_argument(
        "--prep-backprop",
        action="store_true",
        help=(
            "Backpropagate gradients through preprocessing during gradient "
            "approximation step (only works with HSJ, QEBA)."
        ),
    )
    parser.add_argument(
        "--prep-proj",
        action="store_true",
        help=("Project after each attack update (only works with HSJ, QEBA)."),
    )
    parser.add_argument(
        "--smart-noise",
        action="store_true",
        help="Enable Smart Noise Sampling (SNS) from Gao et al. [ICML 2022].",
    )
    # ============================= Preprocess ============================== #
    # Resize
    parser.add_argument("--resize-out-size", default=224, type=int)
    parser.add_argument("--resize-inv-size", default=None, type=int)
    parser.add_argument("--antialias", action="store_true")
    parser.add_argument("--resize-interp", default="nearest", type=str)
    # Quantize
    parser.add_argument("--quantize-num-bits", default=8, type=int)
    # Crop
    parser.add_argument("--crop-size", default=224, type=int)
    # JPEG
    parser.add_argument("--jpeg-quality", default=99.999, type=float)
    # Neural compression
    parser.add_argument("--neural-model", default=None, type=str)
    parser.add_argument("--neural-quality", default=6, type=int)
    # =============================== Attack ================================ #
    parser.add_argument("--max-iter", default=1000, type=int)
    # RayS
    parser.add_argument("--rays-num-queries", default=1000, type=int)
    # HopSkipJump
    parser.add_argument("--hsj-init-grad-steps", default=100, type=int)
    parser.add_argument("--hsj-max-grad-steps", default=200, type=int)
    parser.add_argument("--hsj-gamma", default=10, type=float)
    parser.add_argument(
        "--hsj-norm-rv",
        action="store_true",
        help=(
            "If True, normalize noise in gradient approximation step (same as "
            "the corrected implementation on Foolbox."
        ),
    )
    # Boundary Attack
    parser.add_argument("--boundary-step", default=1e-2, type=float)
    parser.add_argument("--boundary-orth-step", default=1e-2, type=float)
    # Square Attack
    parser.add_argument("--square-p", default=None, type=float)
    # Bandit Attack
    parser.add_argument("--bandit-fd-eta", default=0.01, type=float)
    parser.add_argument("--bandit-image-lr", default=0.5, type=float)
    # Sign-OPT Attack
    parser.add_argument("--signopt-grad-query", default=200, type=int)
    parser.add_argument("--signopt-grad-bs", default=100, type=int)
    parser.add_argument("--signopt-alpha", default=0.2, type=float)
    parser.add_argument("--signopt-beta", default=0.001, type=float)
    parser.add_argument("--signopt-momentum", default=0, type=float)
    parser.add_argument("--signopt-tgt-init-query", default=0, type=int)
    # Opt Attack
    parser.add_argument("--opt-alpha", default=0.2, type=float)
    parser.add_argument("--opt-beta", default=0.001, type=float)
    # QEBA Attack
    parser.add_argument("--qeba-subspace", default="resize4", type=str)
    parser.add_argument("--qeba-gamma", default=0.01, type=float)
    args = vars(parser.parse_args())
    return args


if __name__ == "__main__":

    args = parse_args()
    os.makedirs("./results", exist_ok=True)

    if args["debug"]:
        args["verbose"] = True

    run_one_setting(args)
