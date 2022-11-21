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

import contextlib
import math
import random

import numpy as np
import torch
import torch.nn as nn
from art.estimators.classification import PyTorchClassifier
from extract_prep.preprocessor import Preprocessor

_EPS = 1e-12


def setup_art(args, model, input_size):
    art_model = PyTorchClassifier(
        model=model,
        clip_values=(0, 1),
        loss=nn.CrossEntropyLoss(),
        input_shape=(3, input_size, input_size),
        nb_classes=args["num_classes"],
    )
    return art_model


def set_random_seed(seed):
    """Set random seed for random, numpy, and torch.
    Args:
        seed (int): random seed to set
    """
    assert isinstance(seed, int)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def set_temp_seed(seed, devices=None):
    """Temporary sets numpy seed within a context."""
    # ====================== Save original random state ===================== #
    rand_state = random.getstate()
    np_state = np.random.get_state()
    # Get GPU devices
    if devices is None:
        num_devices = torch.cuda.device_count()
        devices = list(range(num_devices))
    else:
        devices = list(devices)
    # Save random generator state
    cpu_rng_state = torch.get_rng_state()
    gpu_rng_states = []
    for device in devices:
        gpu_rng_states.append(torch.cuda.get_rng_state(device))
    # Set new seed
    set_random_seed(seed)
    try:
        yield
    finally:
        # Set original random state
        random.setstate(rand_state)
        # Set original numpy state
        np.random.set_state(np_state)
        # Set original torch state
        torch.set_rng_state(cpu_rng_state)
        for device, gpu_rng_state in zip(devices, gpu_rng_states):
            torch.cuda.set_rng_state(gpu_rng_state, device)


def select_targets(model, dataloader, labels):
    """Assume that `model` correctly classifies `images` as `labels`."""
    device = labels.device
    src_images, src_labels = [], []
    dataloader_iterator = iter(dataloader)

    for tgt_label in labels:
        while True:
            # Load one image at a time without creating too many iterators
            # https://github.com/pytorch/pytorch/issues/1917#issuecomment-433698337
            try:
                src_image, _ = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(dataloader)
                src_image, _ = next(dataloader_iterator)

            src_label = model(src_image)[0].argmax()
            if src_label != tgt_label:
                break

        src_images.append(src_image)
        src_labels.append(src_label)

    src_images = torch.cat(src_images, dim=0).to(device)
    src_labels = torch.vstack(src_labels).view_as(labels).to(device)
    return src_images, src_labels


def check_match(x1, x2):
    # NOTE: aim for 1/255 per pixel (or half that)
    return (squared_error(x1, x2, ord="2") < 1e-3).cpu()


def squared_error(x1, x2, ord="2"):
    # TODO: inf
    if ord == "2":
        return ((x1 - x2) ** 2).sum((1, 2, 3))


def overshoot(x_from, x_to, dist):
    """Find a point by moving `x_to` in the direction from `x_from` to `x_to`
    by `dist` distance. `x_from` and `x_to` are assumed to be 4D tensor with
    the first dimension being the batch.
    """
    delta = x_to - x_from
    delta_norm = delta.reshape(delta.size(0), -1).norm(2, 1)
    direction = delta / (delta_norm + _EPS)[:, None, None, None]
    return x_to + dist * direction


def _to_attack_space(x, min_, max_):
    # map from [min_, max_] to [-1, +1]
    a = (min_ + max_) / 2
    b = (max_ - min_) / 2
    x = (x - a) / b

    # from [-1, +1] to approx. (-1, +1)
    x = x * 0.99999

    # from (-1, +1) to (-inf, +inf): atanh(x)
    return 0.5 * torch.log((1 + x) / (1 - x))


def _to_model_space(x, min_, max_):
    """Transforms an input from the attack space to the model space.
    This transformation and the returned gradient are elementwise."""
    # from (-inf, +inf) to (-1, +1)
    x = torch.tanh(x)

    # map from (-1, +1) to (min_, max_)
    a = (min_ + max_) / 2
    b = (max_ - min_) / 2
    x = x * b + a
    return x


def find_nearest_preimage(
    args,
    model,
    y,
    x_orig,
    z_adv,
    preprocess,
    lr=1e-1,
    x_init=None,
    max_epsilon=0,
    init_lambda=1e3,
    num_lambda_steps=10,
    num_opt_steps=3000,
    factor=10,
    criteria="misclassify",
    verbose=False,
):
    if verbose:
        print("Finding pre-image of z_adv...")
    batch_size: int = x_orig.size(0)
    log_steps: int = int(num_opt_steps / 20)

    orig_dtype = x_orig.dtype
    dtype = torch.float32
    prev_lmbda = torch.zeros((batch_size,), device=x_orig.device, dtype=dtype)
    x_orig = x_orig.to(dtype)
    z_adv = z_adv.to(dtype)
    model.to(dtype)

    x_init = x_orig if x_init is None else x_init.to(dtype)
    x_init = x_init.detach()
    best_x_pre = x_init.clone()
    success_idx = torch.zeros_like(prev_lmbda, dtype=torch.bool, device="cpu")
    best_dist = torch.zeros_like(prev_lmbda, device="cpu", dtype=dtype) + 1e20
    max_eps = (0.99999 * max_epsilon) ** 2
    # Normalizing constants for the error terms to remove dimension dependency
    STD_DIM = 224
    scale_dim_x = (STD_DIM / x_orig.shape[-1]) ** 2
    scale_dim_z = (STD_DIM / z_adv.shape[-1]) ** 2

    if args["targeted"]:
        criteria = "targeted"
    condition = {
        "misclassify": lambda x: model(x).argmax(-1) != y,
        "targeted": lambda x: model(x).argmax(-1) == y,
        "dist_to_orig": lambda x: squared_error(x, x_orig) <= max_eps,
        "dist_to_adv": lambda x: check_match(preprocess(x), z_adv),
    }[criteria]

    def success_cond(x):
        return condition(x).cpu()

    # If unsuccessful, whether to increase or decrease lambda
    mode = {
        "misclassify": "increase",
        "targeted": "increase",
        "dist_to_orig": "decrease",
        "dist_to_adv": "increase",
    }[criteria]
    if mode == "decrease":
        factor = 1 / factor
    # TODO: account for decrease mode if needed
    lmbda_hi = prev_lmbda.clone() + init_lambda
    lmbda_lo = prev_lmbda.clone() + init_lambda

    for i in range(num_lambda_steps):

        lmbda = (lmbda_lo + lmbda_hi) / 2
        x_pre = x_init.clone() + torch.randn_like(x_init) * 1e-5
        x_pre.clamp_(1e-6, 1 - 1e-6)
        a_pre = _to_attack_space(x_pre, 0, 1)
        a_pre.requires_grad_()
        # Choice of optimizer also seems to matter here. Some optimizer like
        # AdamW cannot get to very low precision for some reason.
        optimizer = torch.optim.Adam([a_pre], lr=lr, eps=_EPS)
        # SGD is too sensitive to choice of lr, especially as lambda changes
        # optimizer = torch.optim.SGD([a_pre], lr=lr, momentum=0.9, nesterov=True)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.1,
            patience=int(num_opt_steps / 100),
            threshold=1e-3,
            verbose=verbose,
            eps=_EPS,
        )
        best_loss = np.inf

        with torch.enable_grad():
            for j in range(num_opt_steps):
                optimizer.zero_grad()
                x_pre = _to_model_space(a_pre, 0, 1)
                dist_orig = squared_error(x_pre, x_orig, ord=args["ord"])
                dist_orig *= scale_dim_x
                dist_prep = squared_error(preprocess(x_pre), z_adv, ord="2")
                dist_prep *= scale_dim_z
                # This should be sum because we want the learning rate to
                # effectively scale with batch size
                loss = (dist_orig + lmbda**2 * dist_prep).sum()
                loss.backward()
                optimizer.step()
                lr_scheduler.step(loss)
                if loss.isnan().any():
                    raise ValueError("NaN loss encountered!")

                if j % log_steps == 0:
                    if verbose:
                        print(f"step {j:4d}: {loss.item():.6f}")
                    if loss.item() < best_loss * 0.9999:
                        best_loss = loss.item()
                    else:
                        if verbose:
                            print(
                                f"No improvement in {log_steps} steps. "
                                "Stopping..."
                            )
                        break

        x_pre.detach_()
        cur_success_idx = success_cond(x_pre)
        # Have not succeeded including this round (expo search)
        lmbda_lo[~cur_success_idx & ~success_idx] *= factor
        lmbda_hi[~cur_success_idx & ~success_idx] *= factor
        # Have not succeeded before but succeed this round (end of expo search)
        lmbda_lo[cur_success_idx & ~success_idx] = prev_lmbda[
            cur_success_idx & ~success_idx
        ]
        lmbda_hi[cur_success_idx & ~success_idx] = lmbda[
            cur_success_idx & ~success_idx
        ]
        # Have succeeded before but not this round (binary search)
        lmbda_lo[~cur_success_idx & success_idx] = lmbda[
            ~cur_success_idx & success_idx
        ]
        # Have succeeded before and this round (binary search)
        lmbda_hi[cur_success_idx & success_idx] = lmbda[
            cur_success_idx & success_idx
        ]

        # Update preimages that are successful and have small perturbation
        success_idx |= cur_success_idx
        dist = dist_prep if criteria == "dist_to_orig" else dist_orig
        better_dist = cur_success_idx & (dist.cpu() < best_dist)
        best_dist[better_dist] = dist[better_dist].detach().cpu()
        best_x_pre[better_dist] = x_pre[better_dist]
        prev_lmbda = lmbda
        if verbose:
            print(
                "  Distortion (L2 square) in original space: ",
                dist_orig.detach().cpu().numpy(),
            )
            print(
                "  Reconstruction error (L2 square) in processed space: ",
                dist_prep.detach().cpu().numpy(),
            )
            print(
                "  L-inf distance in processed space: ",
                (preprocess(x_pre) - z_adv)
                .reshape(batch_size, -1)
                .abs()
                .max(1)[0]
                .cpu()
                .numpy(),
            )
            print("  lambda: ", lmbda.cpu().numpy())
            print("  cur_success_idx: ", cur_success_idx.cpu().numpy())

    if verbose:
        print(f"=> Final pre-image success: {success_idx.sum()}/{batch_size}")
    model.to(orig_dtype)
    return best_x_pre.detach().to(orig_dtype), success_idx


def expo_search_adv(
    model,
    y,
    x_orig,
    x_adv,
    num_steps=10,
    init_step=None,
    targeted=False,
    factor=10,
    verbose=False,
):
    """
    Exponential search in the adversarial direction to make the attacks succeed
    as many as possible.
    """
    success_idx = model(x_adv).argmax(-1) != y
    if targeted:
        success_idx.logical_not_()
    num_steps_used = torch.zeros_like(y)
    delta = x_adv - x_orig
    delta_norm = delta.reshape(y.size(0), -1).norm(2, 1)[:, None, None, None]
    delta /= delta_norm + _EPS
    x_expo = x_adv.clone()
    new_lo = torch.zeros_like(y, dtype=x_orig.dtype, device=x_orig.device)

    if init_step is None:
        # Initialize step size of exponential search: sqrt(d) * 1e-3
        init_step = math.sqrt(x_adv.numel()) * 1e-3

    for i in range(num_steps):
        step_size = init_step * factor**i
        x = x_adv + delta * step_size
        x.clamp_(0, 1)
        num_steps_used[~success_idx] += 1
        cur_success_idx = model(x).argmax(-1) != y
        if targeted:
            cur_success_idx.logical_not_()
        update_idx = ~success_idx & cur_success_idx
        x_expo[update_idx] = x[update_idx]
        success_idx |= cur_success_idx
        if i == 0:
            lo = delta_norm
        else:
            lo = step_size / factor + delta_norm
        new_lo[update_idx] = (lo / (step_size + delta_norm))[
            update_idx
        ].squeeze()

    if verbose:
        print(f"=> Expo search success: {success_idx.sum()}/{len(y)}")
    return x_expo, num_steps_used, new_lo


def binary_search_best_adv(
    model,
    y,
    x_orig,
    x_adv,
    max_num_steps=10,
    num_steps_used=None,
    lo=None,
    tol=None,
    targeted=False,
    verbose=False,
):
    # TODO: This should not be used with crop
    if num_steps_used is None:
        num_steps_used = torch.zeros_like(y)
    batch_size = x_orig.size(0)
    if lo is None:
        lo = torch.zeros((batch_size, 1, 1, 1), device=x_orig.device)
    else:
        lo = lo[:, None, None, None]
    hi = torch.ones_like(lo)
    best_lmbda = torch.ones_like(lo)
    adv_dist = (x_adv - x_orig).reshape(batch_size, -1).norm(2, 1)
    bs_update_idx = torch.zeros_like(y, dtype=torch.bool)

    for i in range(max_num_steps):
        lmbda = (hi + lo) / 2
        x = lmbda * x_adv + (1 - lmbda) * x_orig
        fail_idx = model(x).argmax(-1) == y
        if targeted:
            fail_idx.logical_not_()
        lo[fail_idx] = lmbda[fail_idx]
        hi[~fail_idx] = lmbda[~fail_idx]
        update_idx = (
            (~fail_idx)
            & (lmbda < best_lmbda).squeeze()
            & (num_steps_used < max_num_steps)
        )
        if tol is not None:
            # Stop updating if binary search is dealing with very small
            # distance. This prevents a bug where non-determinism in PyTorch
            # model results flip the prediction when batch size changes.
            update_idx &= adv_dist / 2 ** (i + 1) > tol
        best_lmbda[update_idx] = lmbda[update_idx]
        num_steps_used += 1
        bs_update_idx |= update_idx
        # DEBUG
        # print(~fail_idx)
        # print(lmbda)
        # print(best_lmbda)
        # print(update_idx)
        # import pdb
        # pdb.set_trace()

    if verbose:
        print(
            f"=> Samples updated by binary search: {bs_update_idx.sum()}/{len(y)}"
        )

    return best_lmbda * x_adv + (1 - best_lmbda) * x_orig


def find_preimage(
    args: dict[str, str | int | float],
    ukp_model: torch.nn.Module,
    kp_model: torch.nn.Module,
    y: torch.Tensor,
    x_orig: torch.Tensor,
    z_adv: torch.Tensor,
    preprocess: Preprocessor,
    verbose: bool = False,
) -> torch.Tensor:
    max_search_steps: int = args["binary_search_steps"]

    if preprocess.has_exact_project:
        # Use exact projection if possible
        x = preprocess.project(z_adv, x_orig)
    elif args["preprocess"] == "resize-crop-quantize":
        resize = preprocess.prep_list[0]
        crop = preprocess.prep_list[1]
        x = crop.project(z_adv, resize.prep(x_orig))
        x = resize.project(x, x_orig)
    else:
        # Otherwise, run optimization-based projection to find the nearest
        # pre-image to z_adv. First, overshoot to reduce instability.
        z_adv_os = overshoot(preprocess.prep(x_orig), z_adv, 1e-2)
        z_adv_os.clamp_(0, 1)
        os_fail = kp_model(z_adv_os).argmax(-1) == y
        if args["targeted"]:
            os_fail.logical_not_()
        z_adv[~os_fail] = z_adv_os[~os_fail]
        max_search_steps -= 1
        x_init = preprocess.atk_to_orig(z_adv)

        if verbose:
            print("=> Running find_nearest_preimage...")
        x, _ = find_nearest_preimage(
            args,
            ukp_model,
            y,
            x_orig,
            z_adv,
            preprocess.prep,
            x_init=x_init,
            max_epsilon=args["epsilon"],
            init_lambda=1e3,
            num_opt_steps=2000,
            factor=10,
            num_lambda_steps=args["lambda_search_steps"],
            criteria="misclassify",
            verbose=verbose,
        )

    # Exponential search solves numerical issue by ensuring sufficient overshoot
    x, num_steps_used, lo = expo_search_adv(
        ukp_model,
        y,
        x_orig,
        x,
        num_steps=max_search_steps,
        init_step=None,
        targeted=args["targeted"],
        verbose=verbose,
    )
    if verbose:
        mean_num_steps = num_steps_used.float().mean()
        print(f"=> Exponential search steps used (mean): {mean_num_steps:.2f}")

    x = binary_search_best_adv(
        ukp_model,
        y,
        x_orig,
        x,
        max_num_steps=max_search_steps,
        num_steps_used=num_steps_used,
        lo=lo,
        tol=1e-3,
        targeted=args["targeted"],
        verbose=verbose,
    )
    return x
