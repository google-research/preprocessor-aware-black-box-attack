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

import os
import pickle
import time

import numpy as np
import scipy.sparse
import torch
import torch.nn.functional as T
from torchvision import transforms

from .base import BasePreprocess
from .util import BICUBIC, BILINEAR, NEAREST, ApplySequence


class Resize(BasePreprocess):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        orig_size = (params["orig_size"], params["orig_size"])
        final_size = (params["resize_out_size"], params["resize_out_size"])
        if params["resize_inv_size"] is None:
            inv_size = final_size
        else:
            inv_size = (params["resize_inv_size"], params["resize_inv_size"])
        antialias = params["antialias"]
        interp = {
            "nearest": NEAREST,
            "bilinear": BILINEAR,
            "bicubic": BICUBIC,
        }[params["resize_interp"]]
        self.interp = params["resize_interp"]

        self.output_size = final_size[0]

        # https://github.com/pytorch/pytorch/pull/64501
        # interp = 'nearest-exact' if params['resize_interp'] == 'nearest' else params['resize_interp']
        # align_corners = False if interp in ['bilinear', 'bicubic'] else None
        # def correct_resize(size):
        #     return lambda x: T.interpolate(x, size=size, mode=interp,
        #                                    align_corners=align_corners,
        #                                    antialias=antialias)
        # self.prep = correct_resize(final_size)
        # self.inv_prep = correct_resize(int_size)
        # self.atk_to_orig = correct_resize(orig_size)

        self.prep = transforms.Resize(
            final_size, interpolation=interp, antialias=antialias
        )
        self.inv_prep = transforms.Resize(
            inv_size, interpolation=interp, antialias=antialias
        )
        self.atk_to_orig = transforms.Resize(
            orig_size, interpolation=interp, antialias=antialias
        )

        self.atk_prep = ApplySequence([self.inv_prep, self.prep])
        self.prepare_atk_img = self.prep

        self.has_exact_project = self.interp in ("nearest", "bilinear")
        if self.interp == "nearest":
            x_temp = torch.arange(orig_size[0] ** 2).view((1,) + orig_size)
            z_temp = self.prep(x_temp)
            self.register_buffer("prep_idx", z_temp.view(-1))
        elif self.interp in ("bilinear", "bicubic"):
            print(
                f'=> Getting linear map for {params["resize_interp"]} resize...'
            )
            mat_name = f'mat_resize_{params["resize_interp"]}_{orig_size[0]}_{final_size[0]}.pkl'
            if os.path.exists(mat_name):
                print("  => Pre-computed mapping found!")
                print(f"  => Loading a mapping from {mat_name}...")
                mat, pinv_mat = pickle.load(open(mat_name, "rb"))
            else:
                print(
                    "  => Computing a linear map from original space to "
                    "resampled space..."
                )
                print(
                    "  => This might take a long time and require lots of "
                    "memory..."
                )
                start_time = time.time()
                batch_size = orig_size[1]
                x_temp = torch.zeros(
                    (
                        batch_size,
                        1,
                    )
                    + orig_size,
                    device="cuda",
                )
                batch_idx = torch.arange(batch_size)
                mat = np.zeros(
                    (final_size[0] ** 2, orig_size[0], orig_size[1]),
                    dtype=np.float32,
                )
                for i in range(orig_size[0]):
                    x_temp[batch_idx, :, i, batch_idx] = 1
                    mat[:, i] = (
                        self.prep(x_temp)
                        .view(batch_size, -1)
                        .permute(1, 0)
                        .cpu()
                        .numpy()
                    )
                    x_temp[batch_idx, :, i, batch_idx] = 0
                mat = mat.reshape((final_size[0] ** 2, orig_size[0] ** 2))
                mat = scipy.sparse.coo_matrix(mat)
                print(
                    f"  => Finished computing mapping in {int(time.time() - start_time)}s"
                )
                start_time = time.time()

                print("  => Computing a pseudo-inverse...")
                # Mx = z, M(x - x0) = z - Mx0, x - x0 = pinv(M) @ (z - Mx0)
                pinv_mat = mat.T @ scipy.sparse.linalg.inv(mat @ mat.T)
                print(
                    f"  => Finished computing inverse in {int(time.time() - start_time)}s"
                )

                print(
                    f"  => Saving the mapping as a sparse matrix at {mat_name}..."
                )
                pickle.dump([mat, pinv_mat], open(mat_name, "wb"))

            # self.mat = torch.from_numpy(mat.toarray())
            # self.pinv_mat = torch.from_numpy(pinv_mat.toarray())
            self.mat = mat.astype(np.float64)
            self.pinv_mat = None
        else:
            raise NotImplementedError(f"Invalid interp mode: {self.interp}.")

    def project(self, z, x):
        batch_size, num_channels, _, _ = z.shape
        x_shape = x.shape
        hx, wx = x_shape[2], x_shape[3]
        if self.interp == "nearest":
            x = x.clone().view(batch_size, num_channels, -1)
            x[:, :, self.prep_idx] = z.view(batch_size, num_channels, -1)
            x = x.reshape(x_shape)
        else:
            if self.pinv_mat is not None:
                z_ = z - self.prep(x)
                delta = (
                    self.pinv_mat[None, None, :, :]
                    * z_.view(batch_size, num_channels, 1, -1).cpu()
                ).sum(-1)
            else:
                # Use iterative algorithm to find inverse of sparse matrix
                delta = x.clone()
                z_ = (z - self.prep(x)).cpu().numpy().astype(np.float64)
                # Iterate over batch
                for i, zi in enumerate(z_):
                    # Iterate over channels
                    for c, zic in enumerate(zi):
                        out = scipy.sparse.linalg.lsmr(
                            self.mat,
                            zic.reshape(-1),
                            atol=0,
                            btol=0,
                            maxiter=1e6,
                            conlim=0,
                        )[0]
                        delta[i, c] = torch.from_numpy(
                            out.astype(np.float32)
                        ).view(hx, wx)
            x = x + delta.view(x_shape).to(z.device)

        x.clamp_(0, 1)
        return x
