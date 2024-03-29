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

import itertools

import numpy as np
import torch
import torch.nn as nn

from .utils import c_table, y_table


class rgb_to_ycbcr_jpeg(nn.Module):
    """Converts RGB image to YCbCr.

    Input:
        image(tensor): batch x 3 x height x width
    Outpput:
        result(tensor): batch x height x width x 3
    """

    def __init__(self):
        super().__init__()
        matrix = np.array(
            [
                [0.299, 0.587, 0.114],
                [-0.168736, -0.331264, 0.5],
                [0.5, -0.418688, -0.081312],
            ],
            dtype=np.float32,
        ).T
        self.register_buffer(
            "shift",
            torch.tensor([0.0, 128.0, 128.0], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "matrix", torch.from_numpy(matrix), persistent=False
        )

    def forward(self, image):
        image = image.permute(0, 2, 3, 1)
        result = torch.tensordot(image, self.matrix, dims=1) + self.shift
        return result


class chroma_subsampling(nn.Module):
    """Chroma subsampling on CbCv channels.

    Input:
        image(tensor): batch x height x width x channel
    Output:
        y(tensor): batch x height x width
        cb(tensor): batch x height/2 x width/2
        cr(tensor): batch x height/2 x width/2
    """

    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(
            kernel_size=2, stride=(2, 2), count_include_pad=False
        )

    def forward(self, image):
        # image_2: [B, C, H, W]
        image_2 = image.permute(0, 3, 1, 2).clone()
        cb = self.avg_pool(image_2[:, 1, :, :].unsqueeze(1))
        cr = self.avg_pool(image_2[:, 2, :, :].unsqueeze(1))
        cb = cb.permute(0, 2, 3, 1)
        cr = cr.permute(0, 2, 3, 1)
        return image[:, :, :, 0], cb.squeeze(3), cr.squeeze(3)


class block_splitting(nn.Module):
    """Splitting image into patches.

    Input:
        image(tensor): batch x height x width
    Output:
        patch(tensor):  batch x h*w/64 x h x w
    """

    def __init__(self):
        super().__init__()
        self.k = 8

    def forward(self, image):
        batch_size, height, _ = image.shape
        image_reshaped = image.view(
            batch_size, height // self.k, self.k, -1, self.k
        )
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.reshape(batch_size, -1, self.k, self.k)


class dct_8x8(nn.Module):
    """Discrete Cosine Transformation.

    Input:
        image(tensor): batch x height x width
    Output:
        dcp(tensor): batch x height x width
    """

    def __init__(self):
        super().__init__()
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
                (2 * y + 1) * v * np.pi / 16
            )
        alpha = np.array([1.0 / np.sqrt(2)] + [1] * 7)
        self.register_buffer(
            "tensor", torch.from_numpy(tensor).float(), persistent=False
        )
        self.register_buffer(
            "scale",
            torch.from_numpy(np.outer(alpha, alpha) * 0.25).float(),
            persistent=False,
        )

    def forward(self, image):
        image = image - 128
        result = self.scale * torch.tensordot(image, self.tensor, dims=2)
        return result


class y_quantize(nn.Module):
    """JPEG Quantization for Y channel
    Input:
        image(tensor): batch x height x width
        rounding(function): rounding function to use
        factor(float): Degree of compression
    Output:
        image(tensor): batch x height x width
    """

    def __init__(self, rounding):
        super().__init__()
        self.rounding = rounding
        self.register_buffer("y_table", y_table, persistent=False)

    def forward(self, image, factor=1.0):
        image = image.float() / (self.y_table * factor)
        image = self.rounding(image)
        return image


class c_quantize(nn.Module):
    """JPEG Quantization for CrCb channels
    Input:
        image(tensor): batch x height x width
        rounding(function): rounding function to use
        factor(float): Degree of compression
    Output:
        image(tensor): batch x height x width
    """

    def __init__(self, rounding):
        super().__init__()
        self.rounding = rounding
        self.register_buffer("c_table", c_table, persistent=False)

    def forward(self, image, factor=1.0):
        image = image.float() / (self.c_table * factor)
        image = self.rounding(image)
        return image


class compress_jpeg(nn.Module):
    """Full JPEG compression algortihm
    Input:
        imgs(tensor): batch x 3 x height x width
        rounding(function): rounding function to use
        factor(float): Compression factor
    Ouput:
        compressed(dict(tensor)): batch x h*w/64 x 8 x 8
    """

    def __init__(self, rounding=torch.round):
        super().__init__()
        self.l1 = nn.Sequential(rgb_to_ycbcr_jpeg(), chroma_subsampling())
        self.l2 = nn.Sequential(block_splitting(), dct_8x8())
        self.c_quantize = c_quantize(rounding=rounding)
        self.y_quantize = y_quantize(rounding=rounding)

    def forward(self, image, factor=1.0):
        y, cb, cr = self.l1(image * 255)
        components = {"y": y, "cb": cb, "cr": cr}
        for k, component in components.items():
            comp = self.l2(component)
            if k in ("cb", "cr"):
                comp = self.c_quantize(comp, factor=factor)
            else:
                comp = self.y_quantize(comp, factor=factor)
            components[k] = comp

        return components["y"], components["cb"], components["cr"]
