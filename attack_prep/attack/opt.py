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

"""Sign-OPT Attack.

Code is adapted from https://github.com/cmhcbb/attackbox/blob/master/attack/OPT_attack.py
"""

import time

import torch

from attack_prep.attack.base import Attack
from attack_prep.attack.sign_opt.sign_opt_util import (
    PytorchModel,
    fine_grained_binary_search,
    fine_grained_binary_search_local,
    normalize,
)


class OptAttack(Attack):
    def __init__(self, model, args, substract_steps=0, **kwargs):
        super().__init__(model, args, **kwargs)
        self.model = PytorchModel(model, bounds=(0, 1))
        self.query_limit = args["opt_max_iter"] - substract_steps
        self.num_directions = 100 if args["ord"] == "2" else 500
        # TODO: we may need a better way for controlling number of queries
        # Can't exactly set max query for line search / binary search
        self.iterations = self.query_limit - self.num_directions
        self.alpha = args["opt_alpha"]  # 0.2
        self.beta = args["opt_beta"]  # 0.001
        self.verbose = args["debug"] or args["verbose"]

    def attack_untargeted(self, x0, y0):
        """Attack the original image and return adversarial example
        model: (pytorch model)
        train_dataset: set of training data
        (x0, y0): original image
        """
        model = self.model
        alpha, beta = self.alpha, self.beta
        if model.predict_label(x0) != y0:
            raise RuntimeError("Fail to classify the image. No need to attack.")

        best_theta, prev_best_theta, g_theta = None, None, float("inf")
        query_count = 0
        self.log(
            "Searching for the initial direction on %d random directions: "
            % (self.num_directions)
        )
        # EDIT: random seed is set as a context outside
        # np.random.seed(0)
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
                    self.log("--------> Found distortion %.4f" % g_theta)

        if g_theta == float("inf"):
            best_theta, g_theta = None, float("inf")
            self.log(
                "Searching for the initial direction on %d random directions: "
                % (self.num_directions)
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
                        self.log("--------> Found distortion %.4f" % g_theta)

        if g_theta == float("inf"):
            self.log("Couldn't find valid initial, failed")
            return x0
        timeend = time.time()
        self.log(
            "==========> Found best distortion %.4f in %.4f seconds using %d queries"
            % (g_theta, timeend - timestart, query_count)
        )

        timestart = time.time()
        g1 = 1.0
        theta, g2 = best_theta, g_theta
        for i in range(self.iterations):
            q = 10
            min_g1 = float("inf")
            gradient = torch.zeros_like(theta)
            u = torch.randn((q,) + theta.shape, device=theta.device)
            u, _ = normalize(u, batch=True)
            ttt = theta.unsqueeze(0) + beta * u
            ttt, _ = normalize(ttt, batch=True)
            for j in range(q):
                g1, count = fine_grained_binary_search_local(
                    model, x0, y0, ttt[j], initial_lbd=g2, tol=beta / 500
                )
                query_count += count
                gradient += (g1 - g2) / beta * u[j]
                if g1 < min_g1:
                    min_g1 = g1
                    min_ttt = ttt[j]
            gradient = 1.0 / q * gradient

            if (i + 1) % 10 == 0:
                dist = (g2 * theta).norm().item()
                self.log(
                    (
                        f"Iteration {i + 1:3d} distortion {dist:.4f} "
                        f"num_queries {query_count}"
                    )
                )
                prev_obj = g2

            min_theta = theta
            min_g2 = g2

            for _ in range(15):
                new_theta = theta - alpha * gradient
                new_theta, _ = normalize(new_theta)
                new_g2, count = fine_grained_binary_search_local(
                    model, x0, y0, new_theta, initial_lbd=min_g2, tol=beta / 500
                )
                query_count += count
                alpha *= 2
                if new_g2 < min_g2:
                    min_theta = new_theta
                    min_g2 = new_g2
                else:
                    break

            if min_g2 >= g2:
                for _ in range(15):
                    alpha *= 0.25
                    new_theta = theta - alpha * gradient
                    new_theta, _ = normalize(new_theta)
                    new_g2, count = fine_grained_binary_search_local(
                        model,
                        x0,
                        y0,
                        new_theta,
                        initial_lbd=min_g2,
                        tol=beta / 500,
                    )
                    query_count += count
                    if new_g2 < g2:
                        min_theta = new_theta
                        min_g2 = new_g2
                        break

            if min_g2 <= min_g1:
                theta, g2 = min_theta, min_g2
            else:
                theta, g2 = min_ttt, min_g1

            if g2 < g_theta:
                best_theta, g_theta = theta, g2

            if alpha < 1e-4:
                alpha = 1.0
                self.log(
                    "Warning: not moving, g2 %lf gtheta %lf" % (g2, g_theta)
                )
                beta *= 0.1
                if beta < 1e-8:
                    break

            # prev_best_theta is kept to make sure that we use the latest theta
            # before max query is reached
            if query_count > self.query_limit:
                break
            prev_best_theta = best_theta.clone()

        target = model.predict_label(x0 + g_theta * prev_best_theta)
        timeend = time.time()
        self.log(
            "\nAdversarial Example Found Successfully: distortion %.4f target %d queries %d \nTime: %.4f seconds"
            % (g_theta, target, query_count, timeend - timestart)
        )

        return x0 + g_theta * prev_best_theta

    def log(self, arg):
        if self.verbose:
            print(arg)

    def run(self, imgs, labels):
        x_adv = torch.zeros_like(imgs)
        for i, (img, label) in enumerate(zip(imgs, labels)):
            x_adv[i] = self.attack_untargeted(img, label)
        return x_adv
