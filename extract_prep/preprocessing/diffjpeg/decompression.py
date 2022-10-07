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

import itertools

import numpy as np
import torch
import torch.nn as nn

from .utils import c_table, y_table


class y_dequantize(nn.Module):
    """ Dequantize Y channel
    Inputs:
        image(tensor): batch x height x width
        factor(float): compression factor
    Outputs:
        image(tensor): batch x height x width
    """

    def __init__(self):
        super(y_dequantize, self).__init__()
        self.register_buffer('y_table', y_table, persistent=False)

    def forward(self, image, factor=1.):
        return image * self.y_table * factor


class c_dequantize(nn.Module):
    """ Dequantize CbCr channel
    Inputs:
        image(tensor): batch x height x width
        factor(float): compression factor
    Outputs:
        image(tensor): batch x height x width
    """

    def __init__(self):
        super(c_dequantize, self).__init__()
        self.register_buffer('c_table', c_table, persistent=False)

    def forward(self, image, factor=1.):
        return image * self.c_table * factor


class idct_8x8(nn.Module):
    """ Inverse discrete Cosine Transformation
    Input:
        dcp(tensor): batch x height x width
    Output:
        image(tensor): batch x height x width
    """

    def __init__(self):
        super(idct_8x8, self).__init__()
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
        self.register_buffer('alpha', torch.from_numpy(np.outer(
            alpha, alpha)).float(), persistent=False)
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos(
                (2 * v + 1) * y * np.pi / 16)
        self.register_buffer('tensor', torch.from_numpy(tensor).float(),
                             persistent=False)

    def forward(self, image):
        image = image * self.alpha
        result = 0.25 * torch.tensordot(image, self.tensor, dims=2) + 128
        return result


class block_merging(nn.Module):
    """ Merge pathces into image
    Inputs:
        patches(tensor) batch x height*width/64, height x width
        height(int)
        width(int)
    Output:
        image(tensor): batch x height x width
    """

    def __init__(self):
        super(block_merging, self).__init__()

    def forward(self, patches, height, width):
        k = 8
        batch_size = patches.shape[0]
        image_reshaped = patches.view(batch_size, height//k, width//k, k, k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.reshape(batch_size, height, width)


class chroma_upsampling(nn.Module):
    """ Upsample chroma layers
    Input: 
        y(tensor): y channel image
        cb(tensor): cb channel
        cr(tensor): cr channel
    Ouput:
        image(tensor): batch x height x width x 3
    """

    def __init__(self):
        super(chroma_upsampling, self).__init__()

    def forward(self, y, cb, cr):
        def repeat(x, k=2):
            height, width = x.shape[1:3]
            x = x.unsqueeze(-1)
            x = x.repeat(1, 1, k, k)
            x = x.view(-1, height * k, width * k)
            return x

        cb = repeat(cb)
        cr = repeat(cr)

        return torch.cat([y.unsqueeze(3), cb.unsqueeze(3), cr.unsqueeze(3)], dim=3)


class ycbcr_to_rgb_jpeg(nn.Module):
    """ Converts YCbCr image to RGB JPEG
    Input:
        image(tensor): batch x height x width x 3
    Outpput:
        result(tensor): batch x 3 x height x width
    """

    def __init__(self):
        super(ycbcr_to_rgb_jpeg, self).__init__()

        matrix = np.array(
            [[1., 0., 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]],
            dtype=np.float32).T
        self.register_buffer(
            'shift', torch.tensor([0., -128., -128.]), persistent=False)
        self.register_buffer(
            'matrix', torch.from_numpy(matrix), persistent=False)

    def forward(self, image):
        result = torch.tensordot(image + self.shift, self.matrix, dims=1)
        return result.permute(0, 3, 1, 2)


class decompress_jpeg(nn.Module):
    """ Full JPEG decompression algortihm
    Input:
        compressed(dict(tensor)): batch x h*w/64 x 8 x 8
        rounding(function): rounding function to use
        factor(float): Compression factor
    Ouput:
        image(tensor): batch x 3 x height x width
    """

    def __init__(self, rounding=torch.round):
        super(decompress_jpeg, self).__init__()
        self.c_dequantize = c_dequantize()
        self.y_dequantize = y_dequantize()
        self.idct = idct_8x8()
        self.merging = block_merging()
        self.chroma = chroma_upsampling()
        self.colors = ycbcr_to_rgb_jpeg()

    def forward(self, y, cb, cr, factor=1., size=None):
        img_height, img_width = size
        components = {'y': y, 'cb': cb, 'cr': cr}
        for k in components.keys():
            if k in ('cb', 'cr'):
                comp = self.c_dequantize(components[k], factor=factor)
                height, width = int(img_height / 2), int(img_width / 2)
            else:
                comp = self.y_dequantize(components[k], factor=factor)
                height, width = img_height, img_width
            comp = self.idct(comp)
            components[k] = self.merging(comp, height, width)
        image = self.chroma(components['y'], components['cb'], components['cr'])
        image = self.colors(image)
        image = torch.min(255 * torch.ones_like(image), torch.max(torch.zeros_like(image), image))
        return image / 255
