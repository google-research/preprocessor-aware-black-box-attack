#Copyright 2022 Google LLC
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

"""
Copied directly and modified syntactically from 
https://github.com/cmhcbb/attackbox/blob/65a82f8ea6beedc1b4339aa05b08443d5c489b8a/attack/Sign_OPT_v2.py
"""
import time

import numpy as np
import torch

from ..base import Attack
from .sign_opt_util import (
    PytorchModel,
    fine_grained_binary_search,
    fine_grained_binary_search_local,
    fine_grained_binary_search_local_targeted,
    fine_grained_binary_search_targeted,
    normalize,
)

start_learning_rate = 1.0


class SignOptAttack(Attack):
    def __init__(
        self,
        model,
        args,
        substract_steps=0,
        targeted_dataloader=None,
        preprocess=None,
        **kwargs,
    ):
        super().__init__(model, args, **kwargs)
        self.model = PytorchModel(model, bounds=(0, 1))
        self.preprocess = preprocess
        self.k = args[
            "signopt_grad_query"
        ]  # Num queries for grad estimate (default: 200)
        self.num_directions = 100
        self.targeted = args["targeted"]
        self.svm = False
        self.query_limit = args["max_iter"] - substract_steps
        self.iterations = int(
            np.ceil((self.query_limit - self.num_directions) / self.k)
        )
        self.momentum = args["signopt_momentum"]  # (default: 0)
        self.alpha = args["signopt_alpha"]  # Update step size (default: 0.2)
        self.beta = args[
            "signopt_beta"
        ]  # Gradient estimate step size (default: 0.001)
        self.verbose = args["debug"]
        self.grad_batch_size = min(args["signopt_grad_bs"], self.k)
        self.tgt_init_query = args["signopt_tgt_init_query"]
        self.targeted_dataloader = targeted_dataloader
        # Unused params
        # self.seed = None
        # self.distortion = None
        # self.stopping = 0.0001
        if self.targeted and self.momentum > 0:
            print("Currently, targeted Sign-OPT does not support momentum!")

    def attack_untargeted(self, x0, y0, starting_point=None, preprocess=None):
        """Attack the original image and return adversarial example
        (x0, y0): original image
        """
        model = self.model
        query_count = 0

        # Calculate a good starting point.
        best_theta, g_theta = None, float("inf")
        if self.verbose:
            print(
                f"Searching for the initial direction on {self.num_directions} random directions: "
            )
        timestart = time.time()
        for i in range(self.num_directions):
            query_count += 1
            theta = torch.randn_like(x0)
            if model.predict_label(x0 + theta) != y0:
                theta, initial_lbd = normalize(theta)
                lbd, count = fine_grained_binary_search(
                    model, x0, y0, theta, initial_lbd, g_theta
                )
                query_count += count
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    if self.verbose:
                        print("--------> Found distortion %.4f" % g_theta)

        timeend = time.time()
        if g_theta == float("inf"):
            return x0, float("inf")
        if self.verbose:
            print(
                "==========> Found best distortion %.4f in %.4f seconds "
                "using %d queries" % (g_theta, timeend - timestart, query_count)
            )

        # Begin Gradient Descent.
        timestart = time.time()
        xg, gg = best_theta, g_theta
        best_pert = gg * xg
        vg = torch.zeros_like(xg)
        alpha, beta = self.alpha, self.beta
        for i in range(self.iterations):
            grad_func = self.sign_grad_svm if self.svm else self.sign_grad_v2
            sign_gradient, grad_queries = grad_func(
                x0, y0, xg, initial_lbd=gg, h=beta
            )

            # Line search
            ls_count = 0
            min_theta = xg
            min_g2 = gg
            min_vg = vg
            for _ in range(15):
                if self.momentum > 0:
                    new_vg = self.momentum * vg - alpha * sign_gradient
                    new_theta = xg + new_vg
                else:
                    new_theta = xg - alpha * sign_gradient
                new_theta, _ = normalize(new_theta)
                new_g2, count = fine_grained_binary_search_local(
                    model, x0, y0, new_theta, initial_lbd=min_g2, tol=beta / 500
                )
                ls_count += count
                alpha *= 2
                if new_g2 < min_g2:
                    min_theta = new_theta
                    min_g2 = new_g2
                    if self.momentum > 0:
                        min_vg = new_vg
                else:
                    break

            if min_g2 >= gg:
                for _ in range(15):
                    alpha *= 0.25
                    if self.momentum > 0:
                        new_vg = self.momentum * vg - alpha * sign_gradient
                        new_theta = xg + new_vg
                    else:
                        new_theta = xg - alpha * sign_gradient
                    new_theta, _ = normalize(new_theta)
                    new_g2, count = fine_grained_binary_search_local(
                        model,
                        x0,
                        y0,
                        new_theta,
                        initial_lbd=min_g2,
                        tol=beta / 500,
                    )
                    ls_count += count
                    if new_g2 < gg:
                        min_theta = new_theta
                        min_g2 = new_g2
                        if self.momentum > 0:
                            min_vg = new_vg
                        break

            if alpha < 1e-4:
                alpha = 1.0
                if self.verbose:
                    print("Warning: not moving")
                beta *= 0.1
                if beta < 1e-8:
                    break

            xg, gg = min_theta, min_g2
            vg = min_vg
            query_count += grad_queries + ls_count

            # EDIT: terminate as soon as max queries are used
            if query_count > self.query_limit:
                break
            best_pert = gg * xg

            if i % 5 == 0 and self.verbose:
                print(
                    "Iteration %3d distortion %.4f num_queries %d"
                    % (i + 1, gg, query_count)
                )

        if self.verbose:
            timeend = time.time()
            target = model.predict_label(x0 + best_pert)
            print(
                "\nAdversarial Example Found Successfully: distortion %.4f target"
                " %d queries %d \nTime: %.4f seconds"
                % (gg, target, query_count, timeend - timestart)
            )

        return (x0 + best_pert).clamp(0, 1), gg

    def sign_grad_v2(self, x0, y0, theta, initial_lbd, h=0.001, D=4):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        K = self.k  # K is Q in the paper (i.e. num queries per grad estimate)
        sign_grad = torch.zeros_like(theta)
        device = x0.device
        batch_size = self.grad_batch_size
        num_batches = int(np.ceil(K / batch_size))
        assert num_batches * batch_size == K
        x0 = x0.unsqueeze(0)
        x_temp = x0 + initial_lbd * theta

        for _ in range(num_batches):
            u = torch.randn(
                (batch_size,) + theta.shape, dtype=theta.dtype, device=device
            )
            u, _ = normalize(u, batch=True)

            sign = torch.ones((batch_size, 1, 1, 1), device=device)
            new_theta = theta + h * u
            new_theta, _ = normalize(new_theta, batch=True)

            # EDIT: apply preprocessing if specified
            x = x0 + initial_lbd * new_theta
            if self.preprocess is not None:
                x = self.preprocess(x.clamp(0, 1))
                u = x - x_temp

            out = self.model.predict_label(x)
            if self.targeted:
                # Targeted case
                sign[out == y0] = -1
            else:
                # Untargeted case
                sign[out != y0] = -1
            sign_grad += (u.sign() * sign).sum(0)

        sign_grad /= K
        return sign_grad, K

    def attack_targeted(self, x0, y0, starting_point=None, preprocess=None):
        """Attack the original image and return adversarial example
        (x0, y0): original image
        """
        model = self.model

        if model.predict_label(x0) == y0:
            print("Image already target. No need to attack.")
            return x0, 0.0

        if self.targeted_dataloader is None:
            print("Need training dataset for initial theta.")
            return x0, 0.0

        # EDIT: Default num_samples is 100 which results in 50k queries even
        # before gradient estimation. It takes ~1000 queries just to find 1
        # candidate point.
        # num_samples = 100
        # print("Searching for the initial direction on %d samples: " % (num_samples))

        best_theta, g_theta = None, float("inf")
        query_count = 0
        ls_total = 0
        sample_count = 0
        timestart = time.time()

        y = model.predict_label(starting_point.cuda())
        # EDIT: Use starting point if they are already successful
        if y == y0:
            theta = starting_point - x0
            theta, initial_lbd = normalize(theta)
            lbd, count = fine_grained_binary_search_targeted(
                model, x0, y0, theta, initial_lbd, g_theta
            )
            query_count += count
            best_theta, g_theta = theta, lbd
        else:
            # Iterate through training dataset. Find best initial point for gradient descent.
            for i, (xi, _) in enumerate(self.targeted_dataloader):

                # EDIT: terminate by num queries instead
                if query_count >= self.tgt_init_query:
                    break
                xi = xi.to(x0.device)

                # EDIT: apply preprocess for kp_attack where model = kp_model
                if preprocess is not None:
                    xi = preprocess(xi)

                batch_size = len(xi)
                yi_pred = model.predict_label(xi)
                query_count += batch_size
                idx_success = yi_pred == y0
                if not idx_success.any():
                    continue
                theta = xi[idx_success] - x0.unsqueeze(0)
                theta, initial_lbd = normalize(theta, batch=True)
                for t, ilbd in zip(theta, initial_lbd):
                    lbd, count = fine_grained_binary_search_targeted(
                        model, x0, y0, t, ilbd, g_theta
                    )
                    query_count += count
                    if lbd < g_theta:
                        best_theta, g_theta = t, lbd
                        if self.verbose:
                            print("--------> Found distortion %.4f" % g_theta)
                sample_count += len(theta)

        timeend = time.time()
        if g_theta == np.inf:
            return x0, float("inf")
        if self.verbose:
            print(
                "==========> Found best distortion %.4f in %.4f seconds using %d queries"
                % (g_theta, timeend - timestart, query_count)
            )

        # Begin Gradient Descent.
        timestart = time.time()
        xg, gg = best_theta, g_theta
        best_pert = gg * xg
        alpha, beta = self.alpha, self.beta
        for i in range(self.iterations):
            sign_gradient, grad_queries = self.sign_grad_v2(
                x0, y0, xg, initial_lbd=gg, h=beta
            )

            # Line search
            ls_count = 0
            min_theta = xg
            min_g2 = gg
            for _ in range(15):
                new_theta = xg - alpha * sign_gradient
                new_theta, _ = normalize(new_theta)
                new_g2, count = fine_grained_binary_search_local_targeted(
                    model, x0, y0, new_theta, initial_lbd=min_g2, tol=beta / 500
                )
                ls_count += count
                alpha = alpha * 2
                if new_g2 < min_g2:
                    min_theta = new_theta
                    min_g2 = new_g2
                else:
                    break

            if min_g2 >= gg:
                for _ in range(15):
                    alpha = alpha * 0.25
                    new_theta = xg - alpha * sign_gradient
                    new_theta, _ = normalize(new_theta)
                    new_g2, count = fine_grained_binary_search_local_targeted(
                        model,
                        x0,
                        y0,
                        new_theta,
                        initial_lbd=min_g2,
                        tol=beta / 500,
                    )
                    ls_count += count
                    if new_g2 < gg:
                        min_theta = new_theta
                        min_g2 = new_g2
                        break

            if alpha < 1e-4:
                alpha = 1.0
                print("Warning: not moving")
                beta *= 0.1
                if beta < 1e-8:
                    break

            xg, gg = min_theta, min_g2
            query_count += grad_queries + ls_count
            ls_total += ls_count

            if query_count > self.query_limit:
                break
            best_pert = gg * xg

            if i % 5 == 0 and self.verbose:
                print(
                    "Iteration %3d distortion %.4f num_queries %d"
                    % (i + 1, gg, query_count)
                )

        adv_target = model.predict_label(x0 + best_pert)
        if adv_target == y0:
            timeend = time.time()
            if self.verbose:
                print(
                    "\nAdversarial Example Found Successfully: distortion %.4f target"
                    " %d queries %d LS queries %d \nTime: %.4f seconds"
                    % (gg, y0, query_count, ls_total, timeend - timestart)
                )
            return (x0 + best_pert).clamp(0, 1), gg
        else:
            print("Failed to find targeted adversarial example.")
            return x0, np.float("inf")

    def __call__(self, input_xi, label_or_target, **kwargs):
        if self.targeted:
            return self.attack_targeted(input_xi, label_or_target, **kwargs)
        return self.attack_untargeted(input_xi, label_or_target, **kwargs)

    def run(self, imgs, labels, tgt=None, **kwargs):
        x_adv = torch.zeros_like(imgs)
        if tgt is None:
            data_iter = zip(imgs, labels)
        else:
            data_iter = zip(imgs, tgt[1], tgt[0])

        for i, data in enumerate(data_iter):
            if tgt is None:
                img, lb, sp = data[0], data[1], None
            else:
                img, lb, sp = data
            x_adv[i] = self(img, lb, starting_point=sp, **kwargs)[0]
        return x_adv


def sign(y):
    """
    y -- numpy array of shape (m,)
    Returns an element-wise indication of the sign of a number.
    The sign function returns -1 if y < 0, 1 if x >= 0. nan is returned for nan inputs.
    """
    y_sign = torch.sign(y)
    y_sign[y_sign == 0] = 1
    return y_sign
