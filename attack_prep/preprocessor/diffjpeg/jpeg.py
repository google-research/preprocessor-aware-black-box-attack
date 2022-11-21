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

"""
This code implements differentiable JPEG compression and is modified from
https://github.com/mlomnitz/DiffJPEG/blob/master/DiffJPEG.py.
Now `quality` can vary between samples in the same batch.
"""

import torch
import torch.nn as nn
from .compression import compress_jpeg
from .decompression import decompress_jpeg
from .utils import diff_round, quality_to_factor


class DiffJPEG(nn.Module):
    def __init__(self, differentiable: bool = True, quality: int = 80) -> None:
        """Initialize the DiffJPEG layer.

        Args:
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme.
        """
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = diff_round
        else:
            rounding = torch.round
        # TODO: Both compress and decompress use a very poor fix on the
        # DataParallel issue: nn.Parameter (or buffer) is not copied to all
        # devices properly.
        self.compress = compress_jpeg(rounding=rounding)
        self.decompress = decompress_jpeg(rounding=rounding)
        assert 0 <= quality <= 100
        self.quality = quality

    def forward(self, x):
        assert x.max() <= 1 and x.min() >= 0
        factor = quality_to_factor(self.quality)
        input_dtype = x.dtype
        y, cb, cr = self.compress(x.float(), factor=factor)
        recovered = self.decompress(y, cb, cr, factor=factor, size=x.shape[2:])
        return recovered.to(input_dtype)
