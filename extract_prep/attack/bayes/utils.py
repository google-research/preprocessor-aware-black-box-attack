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

# pylint: disable=E1101, E0401, E1102
import numpy as np
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms


def proj(pert, eps, inf_norm, discrete):
    # project image into epsilon ball (either L_inf norm or L_2 norm)
    # if discrete=True, project into the boundary of the ball instead of the ball
    if inf_norm:
        if discrete:
            return eps * pert.sign()
        else:
            return pert.clamp(-eps, eps)
    else:
        pert_norm = torch.norm(pert.view(pert.shape[0], -1), dim=1)
        pert_norm = torch.where(
            pert_norm > eps, pert_norm / eps, torch.ones_like(pert_norm)
        )
        return pert / (pert_norm.view(-1, 1, 1, 1) + 1e-6)


def latent_proj(pert, eps):
    # project the latent variables (i.e., FFT variables)
    # into the epsilon L_2 ball
    pert_norm = torch.norm(pert, dim=1) / eps
    return pert.div_(pert_norm.view(-1, 1))


def fft_transform(pert):
    # single channel fft transform
    res = torch.zeros(pert.shape[0], 1, 28, 28)
    t_dim = int(np.sqrt(pert.shape[1]))
    for i in range(pert.shape[0]):
        t = torch.zeros((28, 28, 2))
        t[:t_dim, :t_dim] = pert[i].view(t_dim, t_dim, 2)
        res[i, 0] = torch.fft.ifft(t, 2, norm="ortho")
    return res


def fft_transform_mc(pert, dataset_size, channel, cosine, sine):
    # multi-channel FFT transform (each channel done separately)
    # performs a low frequency perturbation (set of allowable frequencies determined by shape of pert)
    device = pert.device
    res = torch.zeros(
        (pert.shape[0], channel, dataset_size, dataset_size), device=device
    )
    for i in range(pert.shape[0]):
        t = torch.zeros((channel, dataset_size, dataset_size, 2), device=device)
        if cosine and sine:
            t_dim = int(np.sqrt(pert.shape[1] / channel / 2))
            t[:, :t_dim, :t_dim, :] = pert[i].view(channel, t_dim, t_dim, 2)
        elif cosine:
            t_dim = int(np.sqrt(pert.shape[1] / channel))
            t[:, :t_dim, :t_dim, 0] = pert[i].view(channel, t_dim, t_dim)
        elif sine:
            t_dim = int(np.sqrt(pert.shape[1] / channel))
            t[:, :t_dim, :t_dim, 1] = pert[i].view(channel, t_dim, t_dim)
        # EDIT: new torch fft (https://github.com/pytorch/pytorch/issues/49637)
        # import pdb
        # pdb.set_trace()
        # Original code for pytorch 1.4
        # res[i] = torch.irfft(t, 3, normalized=True, onesided=False)
        t = torch.view_as_complex(t)
        # option 1
        out = torch.fft.ifftn(t, dim=(0, 1, 2), norm="ortho")
        res[i] = torch.view_as_real(out)[..., 0]
        # option 2
        # res[i] = torch.fft.irfftn(t, s=t.shape, dim=(0, 1, 2), norm='ortho')
    return res
