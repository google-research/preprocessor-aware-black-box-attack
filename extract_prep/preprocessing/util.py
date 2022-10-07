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

import torch
import torch.nn as nn
from torchvision import transforms

NEAREST = transforms.InterpolationMode.NEAREST
BILINEAR = transforms.InterpolationMode.BILINEAR
BICUBIC = transforms.InterpolationMode.BICUBIC


class ApplySequence(nn.Module):
    def __init__(self, preprocesses):
        super().__init__()
        self.preprocesses = preprocesses

    def forward(self, x):
        for p in self.preprocesses:
            x = p(x)
        return x


class SimpleInterp(object):
    def __init__(self, args, preprocess, alpha=0.1):
        self.args = args
        self.preprocess = preprocess
        self.x_orig = None
        self.alpha = alpha

    def __call__(self, z):
        x = self.preprocess(z)
        x_ = self.alpha * self.x_orig + (1 - self.alpha) * x
        return x_


class RgbToGrayscale(nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)[
            None, :, None, None
        ]
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x):
        return (x * self.weight).sum(1, keepdim=True).expand(-1, 3, -1, -1)
