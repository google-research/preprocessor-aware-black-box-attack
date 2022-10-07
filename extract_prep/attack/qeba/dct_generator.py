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

import numpy as np
import torch
from scipy import fftpack


def get_2d_dct(x):
    return fftpack.dct(fftpack.dct(x.T, norm="ortho").T, norm="ortho")


def get_2d_idct(x):
    return fftpack.idct(fftpack.idct(x.T, norm="ortho").T, norm="ortho")


def RGB_img_dct(img):
    assert len(img.shape) == 3 and img.shape[0] == 3
    signal = np.zeros_like(img)
    for c in range(3):
        signal[c] = get_2d_dct(img[c])
    return signal


def RGB_signal_idct(signal):
    assert len(signal.shape) == 3 and signal.shape[0] == 3
    img = np.zeros_like(signal)
    for c in range(3):
        img[c] = get_2d_idct(signal[c])
    return img


class DCTGenerator:
    def __init__(self, factor, batch_size=32, preprocess=None):
        self.factor = factor
        self.batch_size = batch_size
        self.preprocess = preprocess

    def generate_ps(self, inp, N, level=None):
        # print(inp.shape) # (218, 178, 3)
        if self.preprocess is not None:
            transp, mean, std = self.preprocess
            inp = inp.transpose(*transp)
            inp = (inp - mean) / std
        # print(inp.shape) # (3, 218, 178)

        C, H, W = inp.shape
        h_use, w_use = int(H / self.factor), int(W / self.factor)

        ps = []
        for _ in range(N):
            p_signal = torch.zeros_like(inp)
            for c in range(C):
                rv = torch.randn((h_use, w_use))
                rv_ortho, _ = torch.linalg.qr(rv, mode="complete")
                p_signal[c, :h_use, :w_use] = rv_ortho
            p_img = RGB_signal_idct(p_signal.cpu().numpy())
            ps.append(p_img)
        ps = torch.from_numpy(np.stack(ps, axis=0), device=inp.device)

        if self.preprocess is not None:
            rev_transp = np.argsort(transp)
            ps = ps * std
            ps = ps.transpose(0, *(rev_transp + 1))
        # print(ps.shape) # (100, 218, 178, 3)
        return ps

    def calc_rho(self, gt, inp):
        # print(inp.shape) # (218, 178, 3)
        if self.preprocess is not None and inp.shape[2] == 3:
            transp, mean, std = self.preprocess
            inp = inp.transpose(*transp)
            gt = gt.transpose(*transp)

        C, H, W = inp.shape
        h_use, w_use = int(H / self.factor), int(W / self.factor)

        gt_signal = RGB_img_dct(gt)
        rho = np.sqrt(
            (gt_signal[:h_use, :w_use] ** 2).sum() / (gt_signal**2).sum()
        )
        return rho
