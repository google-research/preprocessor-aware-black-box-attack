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

"""Implementation of QEBA attack (GPU version)."""

from __future__ import annotations, division, print_function

import math
import time
import warnings
from typing import Callable

import numpy as np
import torch

from attack_prep.attack.base import Attack
from attack_prep.attack.qeba.util import load_pgen

_EPS = 1e-9


class QEBA(Attack):
    """QEBA attack."""

    def __init__(
        self,
        model: torch.nn.Module,
        args: dict[str, str | int | float],
        substract_steps: int = 0,
        preprocess: Callable[[torch.Tensor], torch.Tensor] | None = None,
        prep_backprop: bool = False,
        smart_noise: Callable[..., np.ndarray] | None = None,
        **kwargs,
    ):
        super().__init__(model, args, **kwargs)
        self.model = model
        self.preprocess = preprocess
        self.max_num_query = args["max_iter"] - substract_steps
        self.initial_num_evals = 100
        self.max_num_evals = 100
        self.stepsize_search = "geometric_progression"
        self.gamma = args["qeba_gamma"]  # Step size (default: 0.01)
        self.batch_size = 50
        self.verbose = args["debug"] or args["verbose"]
        self.log_every_n_steps = 10
        self.iterations = 64  # TODO: may be larger?
        self.plot_adv = False
        self.internal_dtype = torch.float64
        self.constraint = f'l{args["ord"]}'
        self.atk_level = 999
        self.external_dtype = None
        self.rv_generator = load_pgen(args["qeba_subspace"])
        self.prep_backprop = prep_backprop
        if prep_backprop:
            assert preprocess is not None
        self.smart_noise = smart_noise

    def run(self, imgs, labels, tgt=None, **kwargs):
        if tgt is None:
            raise RuntimeError(
                "QEBA is a targeted attack. tgt argument must be specified."
            )
        x_adv = imgs.clone()
        src_images, src_labels = tgt
        for i, _ in enumerate(src_images):
            x_adv[i] = self._attack(imgs[i], src_images[i], src_labels[i])
        return x_adv

    def _gen_random_basis(self, N, device):
        basis = torch.randn((N, *self.shape), device=device)
        return basis

    def _gen_custom_basis(self, N, sample):
        if self.rv_generator is not None:
            basis = self.rv_generator.generate_ps(sample, N, self.atk_level)
        else:
            basis = self._gen_random_basis(N, sample.device)
        return basis.to(self.internal_dtype)

    def _attack(self, img, src_img, src_label):
        self.external_dtype = img.dtype
        self.logger = []

        # Set binary search threshold.
        self.shape = img.shape
        self.fourier_basis_aux = None
        self.d = np.prod(self.shape)
        if self.constraint == "l2":
            self.theta = self.gamma / np.sqrt(self.d)
        else:
            self.theta = self.gamma / (self.d)

        self._printv("QEBA optimized for {} distance".format(self.constraint))
        self.t_initial = time.time()

        # ===========================================================
        # intialize time measurements
        # ===========================================================
        self.time_gradient_estimation = 0
        self.time_search = 0
        self.time_initialization = 0
        self.num_query = 0

        # ===========================================================
        # Construct batch decision function with binary output.
        # ===========================================================
        def decision_function(x):
            if x.ndim == 3:
                x = x.unsqueeze(0)
            assert x.ndim == 4
            outs = torch.zeros(len(x), device=x.device)
            num_batchs = int(math.ceil(len(x) * 1.0 / self.batch_size))
            for j in range(num_batchs):
                current_batch = x[
                    self.batch_size * j : self.batch_size * (j + 1)
                ]
                current_batch = current_batch.to(self.external_dtype)
                outs[self.batch_size * j : self.batch_size * (j + 1)] = (
                    self.model(current_batch.clamp(0, 1)).argmax(1) == src_label
                )
            self.num_query += len(x)
            return outs

        # ===========================================================
        # Initialize variables, constants, hyperparameters, etc.
        # ===========================================================

        # make sure repeated warnings are shown
        warnings.simplefilter("always", UserWarning)

        # get bounds
        self.clip_min, self.clip_max = 0, 1

        # ===========================================================
        # Find starting point
        # ===========================================================
        self.time_initialization += time.time() - self.t_initial

        original = img.to(self.internal_dtype)
        perturbed = src_img.to(self.internal_dtype)

        # ===========================================================
        # Iteratively refine adversarial
        # ===========================================================
        t0 = time.time()

        # Project the initialization to the boundary.
        perturbed, dist_post_update, mask_succeed = self._binary_search_batch(
            original, perturbed.unsqueeze(0), decision_function
        )

        dist = self._compute_distance(perturbed, original)
        self.time_search += time.time() - t0

        if mask_succeed > 0:
            return

        prev_best_perturbed = None

        # Decision boundary direction
        for step in range(1, self.iterations + 1):

            t0 = time.time()
            # ===========================================================
            # Gradient direction estimation.
            # ===========================================================
            # Choose delta.
            delta = self._select_delta(dist_post_update, step)

            # Choose number of evaluations.
            num_evals = int(
                min(
                    [self.initial_num_evals * np.sqrt(step), self.max_num_evals]
                )
            )

            # approximate gradient.
            gradf, _ = self._approximate_gradient(
                decision_function, perturbed, num_evals, delta
            )

            # Calculate auxiliary information for the exp
            # dist_dir = original - perturbed
            # rho = 1.0
            update = gradf.sign() if self.constraint == "linf" else gradf
            t1 = time.time()
            self.time_gradient_estimation += t1 - t0

            # ===========================================================
            # Update, and binary search back to the boundary.
            # ===========================================================
            if self.stepsize_search == "geometric_progression":
                # find step size.
                epsilon = self._geometric_progression_for_stepsize(
                    perturbed, update, dist, decision_function, step
                )

                # Update the sample.
                perturbed += (epsilon * update).to(self.internal_dtype)
                perturbed.clamp_(self.clip_min, self.clip_max)

                # Binary search to return to the boundary.
                (
                    perturbed,
                    dist_post_update,
                    mask_succeed,
                ) = self._binary_search_batch(
                    original, perturbed[None], decision_function
                )

            elif self.stepsize_search == "grid_search":
                # Grid search for stepsize.
                epsilons = np.logspace(-4, 0, num=20, endpoint=True) * dist
                epsilons_shape = [20] + len(self.shape) * [1]
                perturbeds = (
                    perturbed + epsilons.reshape(epsilons_shape) * update
                )
                perturbeds.clamp_(self.clip_min, self.clip_max)
                idx_perturbed = decision_function(perturbeds)

                if np.sum(idx_perturbed) > 0:
                    # Select the perturbation that yields the minimum distance
                    # after binary search.
                    (
                        perturbed,
                        dist_post_update,
                        mask_succeed,
                    ) = self._binary_search_batch(
                        original, perturbeds[idx_perturbed], decision_function
                    )
            t2 = time.time()
            self.time_search += t2 - t1

            if self.num_query >= self.max_num_query:
                break
            prev_best_perturbed = perturbed.clone()

            # compute new distance.
            dist = self._compute_distance(perturbed, original)

            # ===========================================================
            # Log the step
            # ===========================================================
            message = " (took {:.5f} seconds)".format(t2 - t0)
            # self.log_step(step, distance, message, a=a, perturbed=perturbed,
            #               update=update*epsilon, aux_info=(gradf, None, dist_dir, rho))
            # self.printv("Call in grad approx / geo progress / binary search: %d/%d/%d" % (c1-c0, c2-c1, c3-c2))
            if step % self.log_every_n_steps == 0:
                self._printv("Step {}: {:.5e} {}".format(step, dist, message))

            if mask_succeed > 0:
                break

        # ===========================================================
        # Log overall runtime
        # ===========================================================
        self._log_time()
        return prev_best_perturbed

    def _compute_distance(self, x1, x2):
        use_batch = (x1.ndim > 3) or (x2.ndim > 3)
        if x1.ndim == 3:
            x1 = x1.unsqueeze(0)
        if x2.ndim == 3:
            x2 = x2.unsqueeze(0)
        assert x1.ndim == x2.ndim

        diff = x1 - x2
        batch_size = len(diff)
        if self.constraint == "l2":
            dist = diff.reshape(batch_size, -1).norm(2, 1)
        elif self.constraint == "linf":
            dist = diff.reshape(batch_size, -1).abs().max(1)[0]
        return dist if use_batch else dist[0]

    def _project(self, unperturbed, perturbed_inputs, alphas):
        """Projection onto given l2 / linf balls in a batch."""
        alphas_shape = [len(alphas)] + [1] * len(self.shape)
        alphas = alphas.reshape(alphas_shape)
        if self.constraint == "l2":
            projected = (1 - alphas) * unperturbed + alphas * perturbed_inputs
        elif self.constraint == "linf":
            projected = torch.min(
                torch.max(perturbed_inputs, unperturbed - alphas),
                unperturbed + alphas,
            )
        return projected

    def _binary_search_batch(
        self, unperturbed, perturbed_inputs, decision_function
    ):
        """Binary search to approach the boundary."""
        device = perturbed_inputs.device

        # Compute distance between each of perturbed and unperturbed input.
        dists_post_update = self._compute_distance(
            unperturbed, perturbed_inputs
        )

        # Choose upper thresholds in binary searchs based on constraint.
        if self.constraint == "linf":
            highs = dists_post_update
            # Stopping criteria.
            thresholds = dists_post_update * self.theta
            thresholds.clamp_max_(self.theta)
        else:
            highs = torch.ones(
                len(perturbed_inputs), device=device, dtype=self.internal_dtype
            )
            thresholds = self.theta
        lows = torch.zeros_like(highs)

        # Call recursive function.
        while torch.max((highs - lows) / thresholds) > 1:
            # projection to mids.
            mids = (highs + lows) / 2.0
            mid_inputs = self._project(unperturbed, perturbed_inputs, mids)

            # Update highs and lows based on model decisions.
            decisions = decision_function(mid_inputs)
            lows = torch.where(decisions == 0, mids, lows)
            highs = torch.where(decisions == 1, mids, highs)

        out_inputs = self._project(unperturbed, perturbed_inputs, highs)

        # Compute distance of the output to select the best choice.
        # (only used when stepsize_search is grid_search.)
        dists = self._compute_distance(unperturbed, out_inputs)
        idx = dists.argmin()

        dist = dists_post_update[idx]
        out = out_inputs[idx]
        return out, dist, False

    def _select_delta(self, dist_post_update, current_iteration):
        """Choose delta at scale of distance between x and perturbed sample."""
        if current_iteration == 1:
            delta = 0.1 * (self.clip_max - self.clip_min)
        else:
            if self.constraint == "l2":
                delta = np.sqrt(self.d) * self.theta * dist_post_update
            elif self.constraint == "linf":
                delta = self.d * self.theta * dist_post_update

        return delta

    def _approximate_gradient(
        self, decision_function, sample, num_evals, delta
    ):
        """Gradient direction estimation."""
        axis = tuple(range(1, 1 + len(self.shape)))

        if self.smart_noise is not None:
            rv = self.smart_noise(sample, num_evals)
            rv = torch.from_numpy(rv).to(sample.device)
        else:
            rv = self._gen_custom_basis(num_evals, sample)

        # rv /= (rv ** 2).sum(dim=axis, keepdim=True).sqrt() + _EPS
        norm_rv = (rv**2).sum(dim=axis, keepdim=True)
        norm_rv.sqrt_()
        norm_rv.clamp_min_(_EPS)
        rv.div_(norm_rv)

        # perturbed = sample + delta * rv
        rv.mul_(delta)
        rv.add_(sample)
        perturbed = rv
        perturbed.clamp_(self.clip_min, self.clip_max)

        # EDIT: apply preprocess if specified
        if self.preprocess is not None:
            perturbed = self.preprocess(perturbed)
            temp_sample = sample.unsqueeze(0)
            if self.prep_backprop:
                with torch.enable_grad():
                    temp_sample.requires_grad_()
                    out = self.preprocess(temp_sample)
            else:
                out = self.preprocess(temp_sample)
        sample = out.squeeze(0).detach()

        # rv = (perturbed - sample) / delta
        rv = perturbed - sample
        rv /= delta

        # query the model.
        decisions = decision_function(perturbed)
        decision_shape = [len(decisions)] + [1] * len(self.shape)
        fval = (
            2 * decisions.to(self.internal_dtype).reshape(decision_shape) - 1.0
        )

        # Baseline subtraction (when fval differs)
        vals = fval if fval.mean().abs() == 1.0 else fval - fval.mean()
        # gradf = torch.mean(vals * rv, dim=0)
        rv.mul_(vals)
        gradf = rv.mean(dim=0)

        # Backprop gradient through the preprocessor
        if self.preprocess is not None and self.prep_backprop:
            with torch.enable_grad():
                gradf.unsqueeze_(0)
                out.backward(gradf)
                gradf = temp_sample.grad
                gradf.detach_()
                gradf.squeeze_(0)

        # Get the gradient direction.
        gradf /= gradf.norm() + _EPS
        return gradf, fval.mean()

    def _geometric_progression_for_stepsize(
        self, x, update, dist, decision_function, current_iteration
    ):
        """Geometric progression to search for stepsize.

        Keep decreasing stepsize by half until reaching the desired side of the
        boundary.
        """
        epsilon = dist / np.sqrt(current_iteration)
        while True:
            updated = x + epsilon * update
            updated.clamp_(self.clip_min, self.clip_max)
            updated.unsqueeze_(0)
            success = decision_function(updated)
            success.squeeze_(0)
            if success:
                break
            epsilon /= 2

        return epsilon

    # def log_step(self, step, distance, message='', always=False, a=None,
    #              perturbed=None, update=None, aux_info=None):
    #     if not self.verbose:
    #         return
    #     assert len(self.logger) == step
    #     if aux_info is not None:
    #         gradf, grad_gt, dist_dir, rho = aux_info
    #         cos_est = cos_sim(-gradf, grad_gt)
    #         cos_distpred = cos_sim(dist_dir, -gradf)
    #         cos_distgt = cos_sim(dist_dir, grad_gt)

    #         self.logger.append((a._total_prediction_calls, distance, cos_est.item(),
    #                            rho, cos_distpred.item(), cos_distgt.item()))
    #         #cos1 = cos_sim(gradf, grad_gt)
    #         #rand = np.random.randn(*gradf.shape)
    #         #cos2 = cos_sim(grad_gt, rand)
    #         # print ("# evals: %.6f; with gt: %.6f; random with gt: %.6f"%(num_evals, cos1, cos2))
    #         #print ("\testiamted with dist: %.6f; gt with dist: %.6f"%(cos_sim(gradf, original-perturbed), cos_sim(grad_gt, original-perturbed)))
    #     else:
    #         self.logger.append((a._total_prediction_calls, distance, 0, 0, 0, 0))
    #     if not always and step % self.log_every_n_steps != 0:
    #         return

    #     if aux_info is not None:
    #         self.printv("\tEstimated vs. GT: %.6f" % cos_est)
    #         self.printv("\tRho: %.6f" % rho)
    #         self.printv("\tEstimated vs. Distance: %.6f" % cos_distpred)
    #         self.printv("\tGT vs. Distance: %.6f" % cos_distgt)

    def _log_time(self):
        if not self.verbose:
            return
        t_total = time.time() - self.t_initial
        rel_initialization = self.time_initialization / t_total
        rel_gradient_estimation = self.time_gradient_estimation / t_total
        rel_search = self.time_search / t_total

        self._printv("Time since beginning: {:.5f}".format(t_total))
        self._printv(
            "   {:2.1f}% for initialization ({:.5f})".format(
                rel_initialization * 100, self.time_initialization
            )
        )
        self._printv(
            "   {:2.1f}% for gradient estimation ({:.5f})".format(
                rel_gradient_estimation * 100, self.time_gradient_estimation
            )
        )
        self._printv(
            "   {:2.1f}% for search ({:.5f})".format(
                rel_search * 100, self.time_search
            )
        )

    def _printv(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
