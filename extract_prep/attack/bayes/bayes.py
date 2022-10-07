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
Code is adapted from https://github.com/satyanshukla/bayes_attack
"""
import warnings

import torch
import torch.nn as nn

# EDIT: some imports are changed for compatibility with newer BoTorch
from botorch.acquisition import (
    ExpectedImprovement,
    PosteriorMean,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
    qExpectedImprovement,
)
from botorch.fit import fit_gpytorch_model
from botorch.generation import gen_candidates_torch, get_best_candidates
from botorch.models import SingleTaskGP
from botorch.optim import gen_batch_initial_conditions, optimize_acqf
from botorch.sampling.samplers import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood

from ..base import Attack
from .utils import fft_transform_mc, latent_proj, proj

warnings.filterwarnings("ignore")


class ClipModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.cnn_model = model

    def predict_scores(self, x):
        x = x.clamp(0, 1)
        return self.cnn_model(x)


class BayesOptAttack(Attack):
    def __init__(self, model, args, substract_steps=0, **kwargs):
        super().__init__(model, args, **kwargs)
        self.cnn_model = ClipModel(model)
        self.initial_samples = 5  # number of samples taken to form the GP prior
        self.q = 1  # number of candidates to receive from acquisition function
        self.iter = int(
            (args["bayes_max_iter"] - substract_steps - self.initial_samples)
            / self.q
        )
        self.dim = 12  # dimension of attack
        self.standardize_every_iter = (
            False  # normalize objective values at every BayesOpt iteration
        )
        self.sin = True  # if True, use sine FFT basis vectors
        self.cos = True  # if True, use cosine FFT basis vectors
        self.acqf = "EI"  # BayesOpt acquisition function
        self.standardize = False  # normalize objective values
        self.beta = 1.0  # hyperparam for UCB acquisition function
        # if True, project to boundary of epsilon ball (instead of just projecting inside)
        self.discrete = False
        self.hard_label = True  # TODO

        # Fixed params
        self.seed = args["seed"]
        self.eps = args["epsilon"]
        self.inf_norm = args["ord"] == "inf"
        self.channel = 3
        self.arch = "resnet50"
        self.verbose = args["debug"]
        self.device = "cuda"
        # backend for acquisition function optimization (torch or scipy)
        self.optimize_acq = "torch"
        # hyperparam for acquisition function
        self.num_restarts = 1

        if self.sin and self.cos:
            self.latent_dim = self.dim * self.dim * self.channel * 2
        else:
            self.latent_dim = self.dim * self.dim * self.channel

        self.bounds = torch.tensor(
            [[-2.0] * self.latent_dim, [2.0] * self.latent_dim],
            device=self.device,
            dtype=torch.float32,
        )

    def run(self, imgs, labels, tgt=None):
        with torch.enable_grad():
            x_adv = torch.zeros_like(imgs)
            for i, (img, label) in enumerate(zip(imgs, labels)):
                x_adv[i] = self.bayes_opt(img, label)
        return x_adv

    def bayes_opt(self, x0, y0):
        """
        Main Bayesian optimization loop. Begins by initializing model, then for each
        iteration, it fits the GP to the data, gets a new point with the acquisition
        function, adds it to the dataset, and exits if it's a successful attack
        """
        best_observed = []
        query_count, success = 0, 0

        # call helper function to initialize model
        (
            train_x,
            train_obj,
            mll,
            model,
            best_value,
            mean,
            std,
        ) = self.initialize_model(x0, y0, n=self.initial_samples)
        if self.standardize_every_iter:
            train_obj = (train_obj - train_obj.mean()) / train_obj.std()
        best_observed.append(best_value)
        query_count += self.initial_samples

        # run self.iter rounds of BayesOpt after the initial random batch
        for i in range(self.iter):

            # fit the model
            fit_gpytorch_model(mll)

            # define the qNEI acquisition module using a QMC sampler
            if self.q != 1:
                qmc_sampler = SobolQMCNormalSampler(
                    num_samples=2000, seed=self.seed
                )
                qEI = qExpectedImprovement(
                    model=model, sampler=qmc_sampler, best_f=best_value
                )
            else:
                if self.acqf == "EI":
                    qEI = ExpectedImprovement(model=model, best_f=best_value)
                elif self.acqf == "PM":
                    qEI = PosteriorMean(model)
                elif self.acqf == "POI":
                    qEI = ProbabilityOfImprovement(model, best_f=best_value)
                elif self.acqf == "UCB":
                    qEI = UpperConfidenceBound(model, beta=self.beta)

            # optimize and get new observation
            new_x, new_obj = self.optimize_acqf_and_get_observation(qEI, x0, y0)

            if self.standardize:
                new_obj = (new_obj - mean) / std

            # update training points
            train_x = torch.cat((train_x, new_x))
            train_obj = torch.cat((train_obj, new_obj))
            if self.standardize_every_iter:
                train_obj = (train_obj - train_obj.mean()) / train_obj.std()

            # update progress
            best_value, best_index = train_obj.max(0)
            best_observed.append(best_value.item())
            best_candidate = train_x[best_index]

            # reinitialize the model so it is ready for fitting on next iteration
            torch.cuda.empty_cache()
            model.set_train_data(train_x, train_obj, strict=False)

            # get objective value of best candidate; if we found an adversary, exit
            best_candidate = best_candidate.view(1, -1)
            best_candidate = fft_transform_mc(
                best_candidate,
                self.input_size,
                self.channel,
                self.cos,
                self.sin,
            )
            best_candidate = proj(
                best_candidate, self.eps, self.inf_norm, self.discrete
            )
            with torch.no_grad():
                adv_label = torch.argmax(
                    self.cnn_model.predict_scores(best_candidate + x0)
                )
            if adv_label != y0:
                success = 1
                if self.inf_norm and self.verbose:
                    print(
                        "Adversarial Label",
                        adv_label.item(),
                        "Norm:",
                        best_candidate.abs().max().item(),
                        f"Num queries: {query_count}",
                    )
                elif self.verbose:
                    print(
                        "Adversarial Label",
                        adv_label.item(),
                        "Norm:",
                        best_candidate.norm().item(),
                        f"Num queries: {query_count}",
                    )
                # return query_count, success
                return best_candidate + x0
            query_count += self.q
        # not successful (ran out of query budget)
        # return query_count, success
        return best_candidate + x0

    def initialize_model(self, x0, y0, n=5):
        """initialize botorch GP model"""
        # generate prior xs and ys for GP
        train_x = (
            2
            * torch.rand(
                n, self.latent_dim, device=self.device, dtype=torch.float32
            )
            - 1
        )
        if not self.inf_norm:
            train_x = latent_proj(train_x, self.eps)
        train_obj = self.obj_func(train_x, x0, y0)
        mean, std = train_obj.mean(), train_obj.std()
        if self.standardize:
            train_obj = (train_obj - train_obj.mean()) / train_obj.std()
        best_observed_value = train_obj.max().item()

        # define models for objective and constraint
        model = SingleTaskGP(train_X=train_x, train_Y=train_obj[:, None])
        model = model.to(train_x)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll = mll.to(train_x)
        return train_x, train_obj, mll, model, best_observed_value, mean, std

    def obj_func(self, x, x0, y0):
        """
        evaluate objective function
        if hard label: -1 if image is correctly classified, 0 otherwise
        (done this way because BayesOpt assumes we want to maximize)
        if soft label, correct logit - highest logit other than correct logit
        in both cases, successful adversarial perturbation iff objective function >= 0
        """
        x = fft_transform_mc(
            x, self.input_size, self.channel, self.cos, self.sin
        )
        x = proj(x, self.eps, self.inf_norm, self.discrete)
        with torch.no_grad():
            y = self.cnn_model.predict_scores(x + x0)

        if not self.hard_label:
            y = torch.log_softmax(y, dim=1)
            max_score = y[:, y0]
            y, index = torch.sort(y, dim=1, descending=True)
            select_index = (index[:, 0] == y0).long()
            next_max = y.gather(1, select_index.view(-1, 1)).squeeze()
            f = torch.max(max_score - next_max, torch.zeros_like(max_score))
        else:
            index = torch.argmax(y, dim=1)
            f = torch.where(
                index == y0, torch.ones_like(index), torch.zeros_like(index)
            ).float()
        return -f

    def optimize_acqf_and_get_observation(self, acq_func, x0, y0):
        # Optimizes the acquisition function, returns new candidate new_x
        # and its objective function value new_obj

        # optimize
        if self.optimize_acq == "scipy":
            candidates = optimize_acqf(
                acq_function=acq_func,
                bounds=self.bounds,
                q=self.q,
                num_restarts=self.num_restarts,
                raw_samples=200,
            )
        else:
            Xinit = gen_batch_initial_conditions(
                acq_func,
                self.bounds,
                q=self.q,
                num_restarts=self.num_restarts,
                raw_samples=500,
            )
            batch_candidates, batch_acq_values = gen_candidates_torch(
                initial_conditions=Xinit,
                acquisition_function=acq_func,
                lower_bounds=self.bounds[0],
                upper_bounds=self.bounds[1],
                verbose=False,
            )
            candidates = get_best_candidates(batch_candidates, batch_acq_values)

        # observe new values
        new_x = candidates.detach()
        if not self.inf_norm:
            new_x = latent_proj(new_x, self.eps)
        new_obj = self.obj_func(new_x, x0, y0)
        return new_x, new_obj
