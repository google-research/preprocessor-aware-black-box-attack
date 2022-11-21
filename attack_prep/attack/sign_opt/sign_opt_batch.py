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
import scipy.spatial
import torch
from numpy import linalg as LA

from ..base import Attack
from .sign_opt_util import PytorchModel

start_learning_rate = 1.0
EPS = 1e-9


def quad_solver(Q, b):
    """
    Solve min_a  0.5*aQa + b^T a s.t. a>=0
    """
    K = Q.shape[0]
    alpha = np.zeros((K,))
    g = b
    Qdiag = np.diag(Q)
    for i in range(20000):
        delta = np.maximum(alpha - g / Qdiag, 0) - alpha
        idx = np.argmax(abs(delta))
        val = delta[idx]
        if abs(val) < 1e-7:
            break
        g = g + val * Q[:, idx]
        alpha[idx] += val
    return alpha


def sign(y):
    """
    y -- numpy array of shape (m,)
    Returns an element-wise indication of the sign of a number.
    The sign function returns -1 if y < 0, 1 if x >= 0. nan is returned for nan inputs.
    """
    y_sign = np.sign(y)
    y_sign[y_sign == 0] = 1
    return y_sign


class SignOptAttack(Attack):
    def __init__(
        self, model, args, substract_steps=0, train_dataset=None, **kwargs
    ):
        super().__init__(model, args, **kwargs)
        self.model = PytorchModel(model, bounds=(0, 1))
        self.k = 200
        self.train_dataset = train_dataset
        self.target = None
        self.distortion = None
        self.seed = None
        self.svm = False
        self.iterations = 1000
        self.query_limit = args["signopt_max_iter"] - substract_steps
        self.momentum = 0.0
        self.stopping = 0.0001
        self.alpha = 0.2
        self.beta = 0.001
        self.verbose = False

    def attack_untargeted(self, x0, y0):
        """Attack the original image and return adversarial example
        model: (pytorch model)
        train_dataset: set of training data
        (x0, y0): original image
        """
        batch_size = x0.size(0)
        model = self.model
        y0 = y0[0]
        query_count = 0
        ls_total = 0

        # EDIT: does not need to check this in batch case
        # if model.predict_label(x0) != y0:
        #     print("Fail to classify the image. No need to attack.")
        #     return x0, 0.0

        # Calculate a good starting point.
        num_directions = 100
        best_theta, g_theta = None, float("inf")
        g_theta = torch.zeros_like(y0, dtype=x0.dtype)
        if self.verbose:
            print(
                f"Searching for the initial direction on {num_directions} random directions: "
            )
        timestart = time.time()
        for i in range(num_directions):
            query_count += 1
            # theta = np.random.randn(*x0.shape)
            theta = torch.randn_like(x0)
            idx = model.predict_label(x0 + theta) != y0
            initial_lbd = (
                theta[idx].view(batch_size, -1).norm(2, 1)[:, None, None, None]
            )
            theta[idx] /= initial_lbd + EPS
            lbd, count = self.fine_grained_binary_search(
                model, x0[idx], y0[idx], theta[idx], initial_lbd, g_theta[idx]
            )
            query_count += count
            if lbd < g_theta:
                best_theta, g_theta = theta, lbd
                if self.verbose:
                    print("--------> Found distortion %.4f" % g_theta)

        timeend = time.time()
        if g_theta == float("inf"):
            return "NA", float("inf")
        if self.verbose:
            print(
                "==========> Found best distortion %.4f in %.4f seconds "
                "using %d queries" % (g_theta, timeend - timestart, query_count)
            )

        # Begin Gradient Descent.
        timestart = time.time()
        xg, gg = best_theta, g_theta
        vg = np.zeros_like(xg)
        distortions = [gg]
        alpha, beta = self.alpha, self.beta
        for i in range(self.iterations):
            if self.svm == True:
                sign_gradient, grad_queries = self.sign_grad_svm(
                    x0, y0, xg, initial_lbd=gg, h=beta
                )
            else:
                sign_gradient, grad_queries = self.sign_grad_v2(
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
                new_theta /= LA.norm(new_theta)
                new_g2, count = self.fine_grained_binary_search_local(
                    model, x0, y0, new_theta, initial_lbd=min_g2, tol=beta / 500
                )
                ls_count += count
                alpha = alpha * 2
                if new_g2 < min_g2:
                    min_theta = new_theta
                    min_g2 = new_g2
                    if self.momentum > 0:
                        min_vg = new_vg
                else:
                    break

            if min_g2 >= gg:
                for _ in range(15):
                    alpha = alpha * 0.25
                    if self.momentum > 0:
                        new_vg = self.momentum * vg - alpha * sign_gradient
                        new_theta = xg + new_vg
                    else:
                        new_theta = xg - alpha * sign_gradient
                    new_theta /= LA.norm(new_theta)
                    new_g2, count = self.fine_grained_binary_search_local(
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
                print("Warning: not moving")
                beta = beta * 0.1
                if beta < 1e-8:
                    break

            xg, gg = min_theta, min_g2
            vg = min_vg

            query_count += grad_queries + ls_count
            ls_total += ls_count
            distortions.append(gg)

            if query_count > self.query_limit:
                break

            if i % 5 == 0:
                print(
                    "Iteration %3d distortion %.4f num_queries %d"
                    % (i + 1, gg, query_count)
                )

        target = model.predict_label(
            x0 + torch.tensor(gg * xg, dtype=torch.float).cuda()
        )
        timeend = time.time()
        print(
            "\nAdversarial Example Found Successfully: distortion %.4f target"
            " %d queries %d \nTime: %.4f seconds"
            % (gg, target, query_count, timeend - timestart)
        )
        return x0 + torch.tensor(gg * xg, dtype=torch.float).cuda(), gg

    def sign_grad_v2(
        self, x0, y0, theta, initial_lbd, h=0.001, D=4, target=None
    ):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        K = self.k
        sign_grad = np.zeros(theta.shape)
        queries = 0
        preds = []
        for iii in range(K):
            u = np.random.randn(*theta.shape)
            u /= LA.norm(u)

            sign = 1
            new_theta = theta + h * u
            new_theta /= LA.norm(new_theta)

            # Targeted case
            if (
                target is not None
                and self.model.predict_label(
                    x0
                    + torch.tensor(
                        initial_lbd * new_theta, dtype=torch.float
                    ).cuda()
                )
                == target
            ):
                sign = -1

            # Untargeted case
            preds.append(
                self.model.predict_label(
                    x0
                    + torch.tensor(
                        initial_lbd * new_theta, dtype=torch.float
                    ).cuda()
                ).item()
            )
            if (
                target is None
                and self.model.predict_label(
                    x0
                    + torch.tensor(
                        initial_lbd * new_theta, dtype=torch.float
                    ).cuda()
                )
                != y0
            ):
                sign = -1
            queries += 1
            sign_grad += np.sign(u) * sign

        sign_grad /= K
        return sign_grad, queries

    def sign_grad_svm(
        self, x0, y0, theta, initial_lbd, h=0.001, K=100, lr=5.0, target=None
    ):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        sign_grad = np.zeros(theta.shape)
        queries = 0
        dim = np.prod(theta.shape)
        X = np.zeros((dim, K))
        for iii in range(K):
            u = np.random.randn(*theta.shape)
            u /= LA.norm(u)

            sign = 1
            new_theta = theta + h * u
            new_theta /= LA.norm(new_theta)

            # Targeted case.
            if (
                target is not None
                and self.model.predict_label(
                    x0
                    + torch.tensor(
                        initial_lbd * new_theta, dtype=torch.float
                    ).cuda()
                )
                == target
            ):
                sign = -1

            # Untargeted case
            if (
                target is None
                and self.model.predict_label(
                    x0
                    + torch.tensor(
                        initial_lbd * new_theta, dtype=torch.float
                    ).cuda()
                )
                != y0
            ):
                sign = -1
            queries += 1
            X[:, iii] = sign * u.reshape((dim,))

        Q = X.transpose().dot(X)
        q = -1 * np.ones((K,))
        G = np.diag(-1 * np.ones((K,)))
        h = np.zeros((K,))
        alpha = quad_solver(Q, q)
        sign_grad = (X.dot(alpha)).reshape(theta.shape)

        return sign_grad, queries

    def fine_grained_binary_search_local(
        self, model, x0, y0, theta, initial_lbd=1.0, tol=1e-5
    ):
        nquery = 0
        lbd = initial_lbd

        if (
            model.predict_label(
                x0 + torch.tensor(lbd * theta, dtype=torch.float).cuda()
            )
            == y0
        ):
            lbd_lo = lbd
            lbd_hi = lbd * 1.01
            nquery += 1
            while (
                model.predict_label(
                    x0 + torch.tensor(lbd_hi * theta, dtype=torch.float).cuda()
                )
                == y0
            ):
                lbd_hi = lbd_hi * 1.01
                nquery += 1
                if lbd_hi > 20:
                    return float("inf"), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd * 0.99
            nquery += 1
            while (
                model.predict_label(
                    x0 + torch.tensor(lbd_lo * theta, dtype=torch.float).cuda()
                )
                != y0
            ):
                lbd_lo = lbd_lo * 0.99
                nquery += 1

        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            nquery += 1
            if (
                model.predict_label(
                    x0 + torch.tensor(lbd_mid * theta, dtype=torch.float).cuda()
                )
                != y0
            ):
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

    def fine_grained_binary_search(
        self, model, x0, y0, theta, initial_lbd, current_best
    ):
        nquery = 0
        if initial_lbd > current_best:
            if (
                model.predict_label(
                    x0
                    + torch.tensor(
                        current_best * theta, dtype=torch.float
                    ).cuda()
                )
                == y0
            ):
                nquery += 1
                return float("inf"), nquery
            lbd = current_best
        else:
            lbd = initial_lbd

        lbd_hi = lbd
        lbd_lo = 0.0

        while (lbd_hi - lbd_lo) > 1e-5:
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            nquery += 1
            if (
                model.predict_label(
                    x0 + torch.tensor(lbd_mid * theta, dtype=torch.float).cuda()
                )
                != y0
            ):
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

    def eval_grad(
        self, model, x0, y0, theta, initial_lbd, tol=1e-5, h=0.001, sign=False
    ):
        fx = initial_lbd  # evaluate function value at original point
        grad = np.zeros_like(theta)
        x = theta
        # iterate over all indexes in x
        it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])

        queries = 0
        while not it.finished:

            # evaluate function at x+h
            ix = it.multi_index
            oldval = x[ix]
            x[ix] = oldval + h  # increment by h
            unit_x = x / LA.norm(x)
            if sign:
                if (
                    model.predict_label(
                        x0
                        + torch.tensor(
                            initial_lbd * unit_x, dtype=torch.float
                        ).cuda()
                    )
                    == y0
                ):
                    g = 1
                else:
                    g = -1
                q1 = 1
            else:
                fxph, q1 = self.fine_grained_binary_search_local(
                    model, x0, y0, unit_x, initial_lbd=initial_lbd, tol=h / 500
                )
                g = (fxph - fx) / (h)

            queries += q1
            x[ix] = oldval  # restore

            # compute the partial derivative with centered formula
            grad[ix] = g
            it.iternext()  # step to next dimension

        return grad, queries

    def attack_targeted(
        self,
        x0,
        y0,
        target,
        alpha=0.2,
        beta=0.001,
        iterations=5000,
        query_limit=40000,
        distortion=None,
        seed=None,
        svm=False,
        stopping=0.0001,
    ):
        """Attack the original image and return adversarial example
        model: (pytorch model)
        train_dataset: set of training data
        (x0, y0): original image
        """
        model = self.model
        y0 = y0[0]
        print(
            "Targeted attack - Source: {0} and Target: {1}".format(
                y0, target.item()
            )
        )

        if model.predict_label(x0) == target:
            print("Image already target. No need to attack.")
            return x0, 0.0

        if self.train_dataset is None:
            print("Need training dataset for initial theta.")
            return x0, 0.0

        if seed is not None:
            np.random.seed(seed)

        num_samples = 100
        best_theta, g_theta = None, float("inf")
        query_count = 0
        ls_total = 0
        sample_count = 0
        print(
            "Searching for the initial direction on %d samples: "
            % (num_samples)
        )
        timestart = time.time()

        # Iterate through training dataset. Find best initial point for gradient descent.
        for i, (xi, yi) in enumerate(self.train_dataset):
            yi_pred = model.predict_label(xi.cuda())
            query_count += 1
            if yi_pred != target:
                continue

            theta = xi.cpu().numpy() - x0.cpu().numpy()
            initial_lbd = LA.norm(theta)
            theta /= initial_lbd
            lbd, count = self.fine_grained_binary_search_targeted(
                model, x0, y0, target, theta, initial_lbd, g_theta
            )
            query_count += count
            if lbd < g_theta:
                best_theta, g_theta = theta, lbd
                print("--------> Found distortion %.4f" % g_theta)

            sample_count += 1
            if sample_count >= num_samples:
                break

            if i > 500:
                break

        timeend = time.time()
        if g_theta == np.inf:
            return x0, float("inf")
        print(
            "==========> Found best distortion %.4f in %.4f seconds using %d queries"
            % (g_theta, timeend - timestart, query_count)
        )

        # Begin Gradient Descent.
        timestart = time.time()
        xg, gg = best_theta, g_theta
        learning_rate = start_learning_rate
        prev_obj = 100000
        distortions = [gg]
        for i in range(iterations):
            if svm == True:
                sign_gradient, grad_queries = self.sign_grad_svm(
                    x0, y0, xg, initial_lbd=gg, h=beta, target=target
                )
            else:
                sign_gradient, grad_queries = self.sign_grad_v2(
                    x0, y0, xg, initial_lbd=gg, h=beta, target=target
                )

            if False:
                # Compare cosine distance with numerical gradient.
                gradient, _ = self.eval_grad(
                    model, x0, y0, xg, initial_lbd=gg, tol=beta / 500, h=0.01
                )
                print(
                    "    Numerical - Sign gradient cosine distance: ",
                    scipy.spatial.distance.cosine(
                        gradient.flatten(), sign_gradient.flatten()
                    ),
                )

            # Line search
            ls_count = 0
            min_theta = xg
            min_g2 = gg
            for _ in range(15):
                new_theta = xg - alpha * sign_gradient
                new_theta /= LA.norm(new_theta)
                new_g2, count = self.fine_grained_binary_search_local_targeted(
                    model,
                    x0,
                    y0,
                    target,
                    new_theta,
                    initial_lbd=min_g2,
                    tol=beta / 500,
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
                    new_theta /= LA.norm(new_theta)
                    (
                        new_g2,
                        count,
                    ) = self.fine_grained_binary_search_local_targeted(
                        model,
                        x0,
                        y0,
                        target,
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
                beta = beta * 0.1
                if beta < 1e-8:
                    break

            xg, gg = min_theta, min_g2

            query_count += grad_queries + ls_count
            ls_total += ls_count
            distortions.append(gg)

            if query_count > query_limit:
                break

            if i % 5 == 0:
                print(
                    "Iteration %3d distortion %.4f num_queries %d"
                    % (i + 1, gg, query_count)
                )

        adv_target = model.predict_label(
            x0 + torch.tensor(gg * xg, dtype=torch.float).cuda()
        )
        if adv_target == target:
            timeend = time.time()
            print(
                "\nAdversarial Example Found Successfully: distortion %.4f target"
                " %d queries %d LS queries %d \nTime: %.4f seconds"
                % (gg, target, query_count, ls_total, timeend - timestart)
            )

            return x0 + torch.tensor(gg * xg, dtype=torch.float).cuda(), gg
        else:
            print("Failed to find targeted adversarial example.")
            return x0, np.float("inf")

    def fine_grained_binary_search_local_targeted(
        self, model, x0, y0, t, theta, initial_lbd=1.0, tol=1e-5
    ):
        nquery = 0
        lbd = initial_lbd

        if (
            model.predict_label(
                x0 + torch.tensor(lbd * theta, dtype=torch.float).cuda()
            )
            != t
        ):
            lbd_lo = lbd
            lbd_hi = lbd * 1.01
            nquery += 1
            while (
                model.predict_label(
                    x0 + torch.tensor(lbd_hi * theta, dtype=torch.float).cuda()
                )
                != t
            ):
                lbd_hi = lbd_hi * 1.01
                nquery += 1
                if lbd_hi > 100:
                    return float("inf"), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd * 0.99
            nquery += 1
            while (
                model.predict_label(
                    x0 + torch.tensor(lbd_lo * theta, dtype=torch.float).cuda()
                )
                == t
            ):
                lbd_lo = lbd_lo * 0.99
                nquery += 1

        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            nquery += 1
            if (
                model.predict_label(
                    x0 + torch.tensor(lbd_mid * theta, dtype=torch.float).cuda()
                )
                == t
            ):
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid

        return lbd_hi, nquery

    def fine_grained_binary_search_targeted(
        self, model, x0, y0, t, theta, initial_lbd, current_best
    ):
        nquery = 0
        if initial_lbd > current_best:
            if (
                model.predict_label(
                    x0
                    + torch.tensor(
                        current_best * theta, dtype=torch.float
                    ).cuda()
                )
                != t
            ):
                nquery += 1
                return float("inf"), nquery
            lbd = current_best
        else:
            lbd = initial_lbd

        lbd_hi = lbd
        lbd_lo = 0.0

        while (lbd_hi - lbd_lo) > 1e-5:
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            nquery += 1
            if (
                model.predict_label(
                    x0 + torch.tensor(lbd_mid * theta, dtype=torch.float).cuda()
                )
                != t
            ):
                lbd_lo = lbd_mid
            else:
                lbd_hi = lbd_mid
        return lbd_hi, nquery

    def __call__(self, input_xi, label_or_target):
        if self.target is not None:
            adv = self.attack_targeted(input_xi, label_or_target)
        else:
            adv = self.attack_untargeted(input_xi, label_or_target)
        return adv

    def run(self, imgs, labels):
        x_adv = self(imgs, labels)[0]
        return x_adv
