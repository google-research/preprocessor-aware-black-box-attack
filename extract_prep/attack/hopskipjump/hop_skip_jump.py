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

"""Adapted from Foolbox official implementation to implement max queries."""

import logging
import math
from typing import Any, Callable, List, Optional, Union

import eagerpy as ep
import numpy as np
import torch
from foolbox.attacks import LinearSearchBlendedUniformNoiseAttack
from foolbox.attacks.base import (
    MinimizationAttack,
    T,
    get_criterion,
    get_is_adversarial,
    raise_if_kwargs,
)
from foolbox.criteria import Criterion
from foolbox.devutils import atleast_kd, flatten
from foolbox.distances import l1, l2, linf
from foolbox.models import Model
from foolbox.tensorboard import TensorBoard
from typing_extensions import Literal


class HopSkipJump(MinimizationAttack):
    """Our modified implementation of HopSkipJump attack.

    A powerful adversarial attack that requires neither gradients nor
    probabilities [#Chen19].

    Args:
        init_attack : Attack to use to find a starting points. Defaults to
            LinearSearchBlendedUniformNoiseAttack. Only used if starting_points
            is None.
        steps : Number of optimization steps within each binary search step.
        initial_gradient_eval_steps: Initial number of evaluations for gradient
            estimation. Larger initial_num_evals increases time efficiency, but
            may decrease query efficiency.
        max_gradient_eval_steps : Maximum number of evaluations for gradient
            estimation.
        stepsize_search : How to search for stepsize; choices are
            'geometric_progression', 'grid_search'. 'geometric progression'
            initializes the stepsize by ||x_t - x||_p / sqrt(iteration), and
            keep decreasing by half until reaching the target side of the
            boundary. 'grid_search' chooses the optimal epsilon over a grid, in
            the scale of ||x_t - x||_p.
        gamma : The binary search threshold theta is gamma / d^1.5 for
                   l2 attack and gamma / d^2 for linf attack.
        tensorboard : The log directory for TensorBoard summaries. If False,
            TensorBoard summaries will be disabled (default). If None, the
            logdir will be runs/CURRENT_DATETIME_HOSTNAME.
        constraint : Norm to minimize, either "l2" or "linf"

    References:
        .. [#Chen19] Jianbo Chen, Michael I. Jordan, Martin J. Wainwright,
        "HopSkipJumpAttack: A Query-Efficient Decision-Based Attack",
        https://arxiv.org/abs/1904.02144
    """

    distance = l1
    _SMALL_NUM: float = 1e-12

    def __init__(
        self,
        max_queries: int = 1000,
        init_attack: Optional[MinimizationAttack] = None,
        steps: int = 64,
        initial_gradient_eval_steps: int = 100,
        max_gradient_eval_steps: int = 10000,
        stepsize_search: Union[
            Literal["geometric_progression"], Literal["grid_search"]
        ] = "geometric_progression",
        gamma: float = 1.0,
        tensorboard: Union[Literal[False], None, str] = False,
        constraint: Union[Literal["linf"], Literal["l2"]] = "l2",
        verbose: bool = False,
        preprocess: Optional[Any] = None,
        prep_backprop: bool = False,
        smart_noise: Optional[Any] = None,
        norm_rv: bool = False,
    ) -> None:
        if init_attack is not None and not isinstance(
            init_attack, MinimizationAttack
        ):
            raise NotImplementedError(
                f"init_attack ({init_attack}) is not a MinimizationAttack."
            )
        self.max_queries = max_queries
        self.init_attack = init_attack
        self.steps = steps
        self.initial_num_evals = initial_gradient_eval_steps
        self.max_num_evals = max_gradient_eval_steps
        self.stepsize_search = stepsize_search
        self.gamma = gamma
        self.tensorboard = tensorboard
        self.constraint = constraint
        self._verbose = verbose
        self.preprocess = preprocess
        self.prep_backprop = prep_backprop
        self.smart_noise = smart_noise
        self._norm_rv: bool = norm_rv

        assert constraint in ("l2", "linf")
        if constraint == "l2":
            self.distance = l2
        else:
            self.distance = linf

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, T],
        *,
        early_stop: Optional[float] = None,
        starting_points: Optional[T] = None,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        originals, restore_type = ep.astensor_(inputs)
        del inputs, kwargs

        criterion = get_criterion(criterion)
        is_adversarial = get_is_adversarial(criterion, model)
        # num_queries = torch.zeros(inputs.size(0), device=inputs.device)
        num_queries = 0
        curr_num_queries: int = 0

        if starting_points is None:
            init_attack: MinimizationAttack
            if self.init_attack is None:
                init_attack = LinearSearchBlendedUniformNoiseAttack(steps=50)
                logging.info(
                    f"Neither starting_points nor init_attack given. Falling"
                    f" back to {init_attack!r} for initialization."
                )
                # EDIT: num queries = 50
                num_queries += 50
            else:
                init_attack = self.init_attack
            # TODO: use call and support all types of attacks (once early_stop
            # is ossible in __call__)
            x_advs = init_attack.run(
                model, originals, criterion, early_stop=early_stop
            )
        else:
            x_advs = ep.astensor(starting_points)

        is_adv = is_adversarial(x_advs)
        # EDIT: checking is_adversarial uses 1 query
        num_queries += 1
        if not is_adv.all():
            failed = is_adv.logical_not().float32().sum()
            if starting_points is None:
                raise ValueError(
                    f"init_attack failed for {failed} of {len(is_adv)} inputs"
                )
            else:
                raise ValueError(
                    f"{failed} of {len(is_adv)} starting_points are not adversarial"
                )
        del starting_points

        tb = TensorBoard(logdir=self.tensorboard)

        # EDIT: (binary search); Project the initialization to the boundary.
        x_advs, nq = self._binary_search(is_adversarial, originals, x_advs)
        num_queries += nq

        # EDIT: don't count query for assert
        assert ep.all(is_adversarial(x_advs))

        distances = self.distance(originals, x_advs)

        for step in range(self.steps):
            delta = self._select_delta(originals, distances, step)

            # Choose number of gradient estimation steps.
            num_gradient_estimation_steps = int(
                min(
                    [
                        self.initial_num_evals * math.sqrt(step + 1),
                        self.max_num_evals,
                    ]
                )
            )

            gradients = self._approximate_gradients(
                is_adversarial, x_advs, num_gradient_estimation_steps, delta
            )
            # EDIT: plus num queries for grad estimates
            num_queries += num_gradient_estimation_steps

            if self.constraint == "linf":
                update = ep.sign(gradients)
            else:
                update = gradients

            if self.stepsize_search == "geometric_progression":
                # find step size.
                epsilons = distances / math.sqrt(step + 1)

                while True:
                    x_advs_proposals = ep.clip(
                        x_advs + atleast_kd(epsilons, x_advs.ndim) * update,
                        0,
                        1,
                    )
                    success = is_adversarial(x_advs_proposals)
                    # EDIT: is_adversarial
                    num_queries += 1
                    epsilons = ep.where(success, epsilons, epsilons / 2.0)

                    if ep.all(success):
                        break

                # Update the sample.
                x_advs = ep.clip(
                    x_advs + atleast_kd(epsilons, x_advs.ndim) * update, 0, 1
                )

                assert ep.all(is_adversarial(x_advs))

                # Binary search to return to the boundary.
                x_advs, nq = self._binary_search(
                    is_adversarial, originals, x_advs
                )
                # EDIT: binary search
                num_queries += nq

                assert ep.all(is_adversarial(x_advs))

            elif self.stepsize_search == "grid_search":
                # Grid search for stepsize.
                epsilons_grid = ep.expand_dims(
                    ep.from_numpy(
                        distances,
                        np.logspace(
                            -4, 0, num=20, endpoint=True, dtype=np.float32
                        ),
                    ),
                    1,
                ) * ep.expand_dims(distances, 0)

                proposals_list = []

                for epsilons in epsilons_grid:
                    x_advs_proposals = (
                        x_advs + atleast_kd(epsilons, update.ndim) * update
                    )
                    x_advs_proposals = ep.clip(x_advs_proposals, 0, 1)

                    mask = is_adversarial(x_advs_proposals)
                    # Count queries for is_adversarial() call
                    num_queries += 1

                    x_advs_proposals, nq = self._binary_search(
                        is_adversarial, originals, x_advs_proposals
                    )
                    # Count queries for binary search
                    num_queries += nq

                    # only use new values where initial guess was already adversarial
                    x_advs_proposals = ep.where(
                        atleast_kd(mask, x_advs.ndim), x_advs_proposals, x_advs
                    )

                    proposals_list.append(x_advs_proposals)

                proposals = ep.stack(proposals_list, 0)
                proposals_distances = self.distance(
                    ep.expand_dims(originals, 0), proposals
                )
                minimal_idx = ep.argmin(proposals_distances, 0)

                x_advs = proposals[minimal_idx]

            distances = self.distance(originals, x_advs)

            # log stats
            tb.histogram("norms", distances, step)

            # EDIT: break when max queries reached
            if num_queries >= self.max_queries:
                if self._verbose:
                    print(
                        f"=> num queries: {curr_num_queries}, num steps: {step}"
                        f", distance: {ep.mean(distances):.3f}"
                    )
                break
            curr_num_queries = num_queries
            best_x_advs = ep.astensor(x_advs.raw.clone())

        return restore_type(best_x_advs)

    def _approximate_gradients(
        self,
        is_adversarial: Callable[[ep.Tensor], ep.Tensor],
        x_advs: ep.Tensor,
        steps: int,
        delta: ep.Tensor,
    ) -> ep.Tensor:
        # (steps, bs, ...)
        noise_shape = tuple([steps] + list(x_advs.shape))

        if self.smart_noise is not None:
            rv = self.smart_noise(x_advs.raw, steps)
            rv = torch.from_numpy(rv).to(x_advs.raw.device)
            rv = ep.expand_dims(ep.astensor(rv), 1)
        else:
            if self.constraint == "l2":
                rv = ep.normal(x_advs, noise_shape)
            elif self.constraint == "linf":
                rv = ep.uniform(x_advs, low=-1, high=1, shape=noise_shape)

        # Fix original bug here with flatten
        rv /= (
            atleast_kd(ep.norms.l2(flatten(rv, keep=2), axis=-1), rv.ndim)
            + self._SMALL_NUM
        )
        delta_ = atleast_kd(ep.expand_dims(delta, 0), rv.ndim)
        scaled_rv = delta_ * rv

        perturbed = x_advs + scaled_rv
        perturbed = ep.clip(perturbed, 0, 1)

        # Apply preprocess in forward pass if specified
        if self.preprocess is not None:
            perturbed = perturbed.raw.squeeze(1)
            perturbed = ep.astensor(self.preprocess(perturbed).unsqueeze(1))
            if self.prep_backprop:
                with torch.enable_grad():
                    x_temp = x_advs.raw
                    x_temp.requires_grad_()
                    out = self.preprocess(x_temp)
                x_advs = ep.astensor(out.detach())
            else:
                x_advs = ep.astensor(self.preprocess(x_advs.raw))

        # Should rv be re-normalized here? It is not in the original
        # implementation, but it is fixed in later commit:
        # https://github.com/bethgelab/foolbox/commit/d11c90585e1b14385dfd2f6777fe3e047ba25089
        rv = (perturbed - x_advs) / (delta_ if self._norm_rv else 2)

        multipliers_list: List[ep.Tensor] = []
        batch_size: int = len(x_advs)
        ones = ep.ones(x_advs, batch_size)
        for step in range(steps):
            decision = is_adversarial(perturbed[step])
            multipliers_list.append(ep.where(decision, ones, -ones))
        # (steps, bs, ...)
        multipliers = ep.stack(multipliers_list, 0)

        mean = ep.mean(multipliers, axis=0, keepdims=True)
        vals = ep.where(
            ep.abs(mean) == 1,
            multipliers,
            # This is variance reduction term (see Eq. (16))
            multipliers - mean,
        )
        grad = ep.mean(atleast_kd(vals, rv.ndim) * rv, axis=0)

        # Backprop gradient through the preprocessor
        if self.preprocess is not None and self.prep_backprop:
            with torch.enable_grad():
                out.backward(grad.raw)
                grad = ep.astensor(x_temp.grad.detach())

        grad /= (
            atleast_kd(ep.norms.l2(flatten(grad, keep=1), axis=-1), grad.ndim)
            + self._SMALL_NUM
        )

        return grad

    def _project(
        self, originals: ep.Tensor, perturbed: ep.Tensor, epsilons: ep.Tensor
    ) -> ep.Tensor:
        """Clips the perturbations to epsilon and returns the new perturbed.

        Args:
            originals: A batch of reference inputs.
            perturbed: A batch of perturbed inputs.
            epsilons: A batch of norm values to project to.

        Returns:
            A tensor like perturbed but with the perturbation clipped to
            epsilon.
        """
        epsilons = atleast_kd(epsilons, originals.ndim)
        if self.constraint == "linf":
            perturbation = perturbed - originals

            # ep.clip does not support tensors as min/max
            clipped_perturbed = ep.where(
                perturbation > epsilons, originals + epsilons, perturbed
            )
            clipped_perturbed = ep.where(
                perturbation < -epsilons,
                originals - epsilons,
                clipped_perturbed,
            )
            return clipped_perturbed
        else:
            return (1.0 - epsilons) * originals + epsilons * perturbed

    def _binary_search(
        self,
        is_adversarial: Callable[[ep.Tensor], ep.Tensor],
        originals: ep.Tensor,
        perturbed: ep.Tensor,
    ) -> ep.Tensor:
        # Choose upper thresholds in binary search based on constraint.
        d = np.prod(perturbed.shape[1:])
        if self.constraint == "linf":
            highs = linf(originals, perturbed)

            # TODO: Check if the threshold is correct
            #  empirically this seems to be too low
            thresholds = highs * self.gamma / (d * d)
        else:
            highs = ep.ones(perturbed, len(perturbed))
            thresholds = self.gamma / (d * math.sqrt(d))

        lows = ep.zeros_like(highs)

        # use this variable to check when mids stays constant and the BS has converged
        old_mids = highs
        # EDIT
        num_queries = 0

        while ep.any(highs - lows > thresholds):
            mids = (lows + highs) / 2
            mids_perturbed = self._project(originals, perturbed, mids)
            is_adversarial_ = is_adversarial(mids_perturbed)
            num_queries += 1

            highs = ep.where(is_adversarial_, mids, highs)
            lows = ep.where(is_adversarial_, lows, mids)

            # check of there is no more progress due to numerical imprecision
            reached_numerical_precision = (old_mids == mids).all()
            old_mids = mids

            if reached_numerical_precision:
                # if num_queries == 20:
                # TODO: warn user
                # FIXME: This is always reached when the loop has been repeated
                # for 26 times (1/2**26 = 1.5e-8), but changing this seems to
                # make the final outcome significantly worse
                # print('bad')
                break
        # print(num_queries)

        res = self._project(originals, perturbed, highs)

        return res, num_queries

    def _select_delta(
        self, originals: ep.Tensor, distances: ep.Tensor, step: int
    ) -> ep.Tensor:
        result: ep.Tensor
        if step == 0:
            result = 0.1 * ep.ones_like(distances)
        else:
            d = np.prod(originals.shape[1:])

            if self.constraint == "linf":
                theta = self.gamma / (d * d)
                result = d * theta * distances
            else:
                theta = self.gamma / (d * np.sqrt(d))
                result = np.sqrt(d) * theta * distances

        return result
