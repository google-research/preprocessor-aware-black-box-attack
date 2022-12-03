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

"""Common attack utilities."""

from __future__ import annotations

import contextlib
import math
import random

import numpy as np
import torch
from art.estimators.classification import PyTorchClassifier
from torch import nn

from attack_prep.preprocessor.base import Preprocessor

_EPS = 1e-12


def setup_art(args, model, input_size):
    """Set up model for ART."""
    art_model = PyTorchClassifier(
        model=model,
        clip_values=(0, 1),
        loss=nn.CrossEntropyLoss(),
        input_shape=(3, input_size, input_size),
        nb_classes=args["num_classes"],
    )
    return art_model


def set_random_seed(seed: int) -> None:
    """Set random seed for random, numpy, and torch.

    Args:
        seed: Random seed to set
    """
    assert isinstance(seed, int)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def set_temp_seed(seed: int, devices=None):
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


def select_targets(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find initial images and labels for targeted attacks.

    Most attacks initialize adversarial examples with another image in test set
    that is classified to a different class from the original. This method finds
    and returns this initial image and its corresponding (predicted) label given
    the original labels.

    Using outputs of this method assumes that model correctly classifies images
    as given labels.

    Args:
        model: Target model to attack.
        dataloader: Test set that we search for the initial samples.
        labels: Original labels of samples to attack.

    Returns
        Tuple of two tensors, target images and their labels.
    """
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


def _check_match(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Check whether x1 and x2 are very close in L2 distance."""
    # NOTE: aim for 1/255 per pixel (or half that)
    return (_squared_error(x1, x2, norm="2") < 1e-3).cpu()


def _squared_error(x1: torch.Tensor, x2: torch.Tensor, norm: str = "2"):
    """Compute distance between x1 and x2."""
    if norm == "2":
        return ((x1 - x2) ** 2).sum((1, 2, 3))
    raise NotImplementedError()


def _overshoot(
    x_from: torch.Tensor, x_to: torch.Tensor, dist: float
) -> torch.Tensor:
    """Overshoot sample sligtly in one direction.

    Find a point by moving `x_to` in the direction from `x_from` to `x_to`
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

    This transformation and the returned gradient are elementwise.
    """
    # from (-inf, +inf) to (-1, +1)
    x = torch.tanh(x)

    # map from (-1, +1) to (min_, max_)
    a = (min_ + max_) / 2
    b = (max_ - min_) / 2
    x = x * b + a
    return x


def _find_nearest_preimage(
    config: dict[str, str | int | float],
    model: nn.Module,
    y: torch.Tesnor,
    x_orig: torch.Tesnor,
    z_adv: torch.Tesnor,
    preprocess: Preprocessor,
    learning_rate: float = 1e-1,
    x_init: torch.Tesnor | None = None,
    init_lambda: float = 1e3,
    num_lambda_steps: int = 10,
    num_opt_steps: int = 3000,
    factor: float = 10.0,
    criterion: str = "misclassify",
    verbose: bool = False,
) -> torch.Tesnor:
    """Find a projection of z_adv that is closest to x_orig in original space.

    This projection (or pre-image) is computed by Adam optimizer minimizing the
    both the distance in the original space and the reconstruction error in the
    processed space (as a Lagrangian). Lagrange multiplier or lambda is
    exponential/binary searched to minimize the objective while making sure that
    the projection is still misclassified.

    Args:
        config: Experiment config.
        model: Target model to attack.
        y: Ground-truth labels.
        x_orig: Original image.
        z_adv: Adversarial examples in the processed space that we want to
            project to the original space.
        preprocess: Preprocessor.
        learning_rate: Step size of the optimizer. Defaults to 1e-1.
        x_init: Point in the original space to initialize the optimization with.
            If preprocessor has an inverse, one can use inverse_prep(z_adv) as
            an initialization. Defaults to None (use x_orig).
        init_lambda: Initial value of lambda. Defaults to 1e3.
        num_lambda_steps: Number of exponential/binary search steps for lambda.
            Each step queries the model once. Defaults to 10.
        num_opt_steps: Number of optimization steps. Defaults to 3000.
        factor: Factor to scale lambda up if projection is not misclassified.
            Defaults to 10.0.
        criterion: Criterion for a successful projection. If criterion is met,
            lambda will be decreased, and then we repeat the optimization.
            Otherwise, we increase lambda. Defaults to "misclassify".
        verbose: If True, progress is printed. Defaults to False.

    Raises:
        ValueError: NaN loss occurs during optimization.

    Returns:
        Pre-image or projection of z_adv.
    """
    if verbose:
        print("Finding pre-image of z_adv...")
    batch_size: int = x_orig.size(0)
    num_logs: int = 20
    log_steps: int = int(num_opt_steps / num_logs)

    orig_dtype = x_orig.dtype
    # We can set higher-precision dtype in case we want better solution
    dtype = orig_dtype  # torch.float32, torch.float64
    prev_lmbda = torch.zeros(batch_size, device=x_orig.device, dtype=dtype)
    x_orig = x_orig.to(dtype)
    z_adv = z_adv.to(dtype)
    model.to(dtype)

    x_init = x_orig if x_init is None else x_init.to(dtype)
    x_init.detach_()
    best_x_pre = x_init.clone()
    success_idx = torch.zeros_like(prev_lmbda, dtype=torch.bool, device="cpu")
    best_dist = torch.zeros_like(prev_lmbda, device="cpu", dtype=dtype) + 1e20
    # Normalizing constants for the error terms to remove dimension dependency
    standard_dim = 224
    scale_dim_x = (standard_dim / x_orig.shape[-1]) ** 2
    scale_dim_z = (standard_dim / z_adv.shape[-1]) ** 2

    if config["targeted"]:
        criterion = "targeted"
    condition = {
        "misclassify": lambda x: model(x).argmax(-1) != y,
        "targeted": lambda x: model(x).argmax(-1) == y,
        "dist_to_adv": lambda x: _check_match(preprocess(x), z_adv),
    }[criterion]

    def success_cond(x):
        return condition(x).cpu()

    lmbda_hi = prev_lmbda.clone() + init_lambda
    lmbda_lo = prev_lmbda.clone() + init_lambda

    for _ in range(num_lambda_steps):

        lmbda = (lmbda_lo + lmbda_hi) / 2
        x_pre = x_init.clone() + torch.randn_like(x_init) * 1e-5
        x_pre.clamp_(1e-6, 1 - 1e-6)
        a_pre = _to_attack_space(x_pre, 0, 1)
        a_pre.requires_grad_()
        # Choice of optimizer also seems to matter here. Some optimizer like
        # AdamW cannot get to very low precision for some reason.
        optimizer = torch.optim.Adam([a_pre], lr=learning_rate, eps=_EPS)
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
        best_loss: float = np.inf
        max_loss: float = -np.inf
        num_loss_inc: int = -1

        with torch.enable_grad():
            for j in range(num_opt_steps):
                optimizer.zero_grad()
                x_pre = _to_model_space(a_pre, 0, 1)
                dist_orig = _squared_error(x_pre, x_orig, norm=config["ord"])
                dist_orig *= scale_dim_x
                dist_prep = _squared_error(preprocess(x_pre), z_adv, norm="2")
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
                    cur_loss = loss.item()
                    if verbose:
                        print(f"step {j:4d}: {cur_loss:.6f}")
                    if cur_loss > max_loss:
                        num_loss_inc += 1
                        max_loss = cur_loss
                    if cur_loss < best_loss * 0.9999:
                        best_loss = cur_loss
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
        dist = dist_prep if criterion == "dist_to_orig" else dist_orig
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

        if num_loss_inc > 0:
            # If loss has increased, then reduce lr by a factor of 10
            learning_rate /= 10

    if verbose:
        print(f"=> Final pre-image success: {success_idx.sum()}/{batch_size}")
    model.to(orig_dtype)
    return best_x_pre.detach().to(orig_dtype), success_idx


def _expo_search_adv(
    model: nn.Module,
    targets: torch.Tensor,
    x_orig: torch.Tensor,
    x_adv: torch.Tensor,
    num_steps: int = 10,
    init_step: int | None = None,
    targeted: bool = False,
    factor: float = 10,
    verbose: bool = False,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """Exponential search in the adversarial direction.

    This deals with numerical instability when inputs lie very close to decision
    boundary. This ensures attacks succeed as much as possible.
    """
    success_idx = model(x_adv).argmax(-1) != targets
    if targeted:
        success_idx.logical_not_()
    num_steps_used = torch.zeros_like(targets)
    delta = x_adv - x_orig
    delta_norm = delta.reshape(targets.size(0), -1).norm(2, dim=1)[
        :, None, None, None
    ]
    delta /= delta_norm + _EPS
    x_expo = x_adv.clone()
    new_lo = torch.zeros_like(targets, dtype=x_orig.dtype, device=x_orig.device)

    if init_step is None:
        # Initialize step size of exponential search: sqrt(d) * 1e-3
        init_step = math.sqrt(x_adv.numel()) * 1e-3

    for i in range(num_steps):
        step_size = init_step * factor**i
        x = x_adv + delta * step_size
        x.clamp_(0, 1)
        num_steps_used[~success_idx] += 1
        cur_success_idx = model(x).argmax(-1) != targets
        if targeted:
            cur_success_idx.logical_not_()
        update_idx = ~success_idx & cur_success_idx
        x_expo[update_idx] = x[update_idx]
        success_idx |= cur_success_idx
        if i == 0:
            low = delta_norm
        else:
            low = step_size / factor + delta_norm
        new_lo[update_idx] = (low / (step_size + delta_norm))[
            update_idx
        ].squeeze()

    if verbose:
        print(f"=> Expo search success: {success_idx.sum()}/{len(targets)}")
    return x_expo, num_steps_used, new_lo


def _binary_search_best_adv(
    model: nn.Module,
    targets: torch.Tensor,
    x_orig: torch.Tensor,
    x_adv: torch.Tensor,
    max_num_steps: int = 10,
    num_steps_used: torch.Tensor | None = None,
    low: torch.Tensor | None = None,
    tol: float | None = None,
    targeted: bool = False,
    verbose: bool = False,
) -> torch.Tensor:
    if num_steps_used is None:
        num_steps_used = torch.zeros_like(targets)
    batch_size = x_orig.size(0)
    if low is None:
        low = torch.zeros((batch_size, 1, 1, 1), device=x_orig.device)
    else:
        low = low[:, None, None, None]
    high = torch.ones_like(low)
    best_lmbda = torch.ones_like(low)
    adv_dist = (x_adv - x_orig).reshape(batch_size, -1).norm(2, 1)
    bs_update_idx = torch.zeros_like(targets, dtype=torch.bool)

    for i in range(max_num_steps):
        lmbda = (high + low) / 2
        x = lmbda * x_adv + (1 - lmbda) * x_orig
        fail_idx = model(x).argmax(-1) == targets
        if targeted:
            fail_idx.logical_not_()
        low[fail_idx] = lmbda[fail_idx]
        high[~fail_idx] = lmbda[~fail_idx]
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

    if verbose:
        print(
            "=> # samples updated by binary search: "
            f"{bs_update_idx.sum()}/{len(targets)}"
        )

    return best_lmbda * x_adv + (1 - best_lmbda) * x_orig


def find_preimage(
    config: dict[str, str | int | float],
    model: torch.nn.Module,
    targets: torch.Tensor,
    x_orig: torch.Tensor,
    z_adv: torch.Tensor,
    preprocess: Preprocessor,
    verbose: bool = False,
) -> torch.Tensor:
    """Recovery step of the Bypassing and the Biased-Gradient Attacks."""
    max_search_steps: int = config["binary_search_steps"]

    if preprocess.has_exact_project:
        # Use exact projection if possible
        print("Using exact projection...")
        x_proj = preprocess.project(z_adv, x_orig)
    elif config["preprocess"] == "resize-crop-quantize":
        resize = preprocess.prep_list[0]
        crop = preprocess.prep_list[1]
        x_proj = crop.project(z_adv, resize.prep(x_orig))
        x_proj = resize.project(x_proj, x_orig)
    else:
        # Otherwise, run optimization-based projection to find the nearest
        # pre-image to z_adv. First, overshoot to reduce instability.
        z_adv_os = _overshoot(preprocess.prep(x_orig), z_adv, 1e-2)
        z_adv_os.clamp_(0, 1)
        os_fail = model(z_adv_os).argmax(-1) == targets
        if config["targeted"]:
            os_fail.logical_not_()
        z_adv[~os_fail] = z_adv_os[~os_fail]
        max_search_steps -= 1
        x_init = preprocess.inv_prep(z_adv)

        if verbose:
            print("=> Running find_nearest_preimage...")
        x_proj, _ = _find_nearest_preimage(
            config,
            model,
            targets,
            x_orig,
            z_adv,
            preprocess.prep,
            learning_rate=1e-1,
            x_init=x_init,
            init_lambda=1e3,
            num_opt_steps=2000,
            factor=10,
            num_lambda_steps=config["lambda_search_steps"],
            criterion="misclassify",
            verbose=verbose,
        )

    # Exponential search solves numerical issue by ensuring sufficient overshoot
    x_proj, num_steps_used, low = _expo_search_adv(
        model,
        targets,
        x_orig,
        x_proj,
        num_steps=max_search_steps,
        init_step=None,
        targeted=config["targeted"],
        verbose=verbose,
    )
    if verbose:
        mean_num_steps = num_steps_used.float().mean()
        print(f"=> Exponential search steps used (mean): {mean_num_steps:.2f}")

    x_proj = _binary_search_best_adv(
        model,
        targets,
        x_orig,
        x_proj,
        max_num_steps=max_search_steps,
        num_steps_used=num_steps_used,
        low=low,
        tol=1e-3,
        targeted=config["targeted"],
        verbose=verbose,
    )
    return x_proj
