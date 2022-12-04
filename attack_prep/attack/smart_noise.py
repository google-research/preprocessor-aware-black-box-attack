"""Smart Noise."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

_ART_NUMPY_DTYPE = np.float32


class SmartNoise(object):
    def __init__(
        self,
        hr_shape: tuple[int, int, int],
        lr_shape: tuple[int, int, int],
        projection: nn.Module | None = None,
        projection_estimate: nn.Module | None = None,
        precise: bool = False,
    ):
        self.hr_shape = hr_shape
        self.lr_shape = lr_shape
        self.projection = projection
        self.projection_estimate = projection_estimate
        self.precise = precise

    @torch.enable_grad()
    def __call__(self, x: np.ndarray, n: int) -> np.ndarray:
        if self.precise:
            return self._get_noise_precise(x, n)
        return self._get_noise_imprecise(x, n)

    def _get_noise_imprecise(self, x: np.ndarray, n: int) -> np.ndarray:
        # setup
        noise = np.zeros((n,) + self.hr_shape, dtype=_ART_NUMPY_DTYPE)
        if len(x.shape) == 4:
            x = x.squeeze(0)
        x_hr = torch.as_tensor(x[None]).cuda()

        # generate n noise
        for i in range(n):
            # generate noise at LR space
            delta_lr = torch.randn(1, *self.lr_shape).cuda()

            # optimizing variable: noise at HR space
            delta_hr = torch.zeros_like(x_hr).requires_grad_()

            # compute perturbed HR and LR images
            perturbed_hr = x_hr + delta_hr
            perturbed_lr = self.projection(x_hr) + delta_lr

            # loss = L2 norm distance at the LR space
            loss = torch.norm(self.projection_estimate(perturbed_hr) - perturbed_lr)
            # loss = torch.norm(self.projection(perturbed_hr) - perturbed_lr)
            loss.backward()

            # select the gradient as the noise
            noise[i] = delta_hr.grad.cpu().numpy().astype(_ART_NUMPY_DTYPE)

        return noise

    def _get_noise_precise(self, x: np.ndarray, n: int) -> np.ndarray:
        # setup
        noise = np.zeros((n,) + self.hr_shape, dtype=_ART_NUMPY_DTYPE)
        x_hr = torch.as_tensor(x[None]).cuda()

        # generate n noise
        for i in range(n):
            # generate noise at LR space
            delta_lr = torch.randn(1, *self.lr_shape).cuda()

            # optimizing variable: noise at HR space
            delta_hr = torch.zeros_like(x_hr).requires_grad_()

            # compute perturbed LR image (same for all optimizations)
            perturbed_lr = self.projection(x_hr) + delta_lr

            # internal minimization
            opt = torch.optim.Adam([delta_hr], lr=0.01)
            for _ in range(1000):
                # loss = L2 norm distance at the LR space
                perturbed_hr = x_hr + delta_hr
                loss = torch.norm(
                    self.projection_estimate(perturbed_hr) - perturbed_lr
                )

                # update HR noise
                opt.zero_grad()
                loss.backward()
                opt.step()

            # select the solution as the noise
            noise[i] = delta_hr.grad.cpu().numpy().astype(_ART_NUMPY_DTYPE)

        return noise
