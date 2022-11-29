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

from torchvision import transforms

from attack_prep.preprocessor.base import Preprocessor
from attack_prep.preprocessor.util import (
    BICUBIC,
    BILINEAR,
    NEAREST,
    ApplySequence,
)


class SimpleInterp:
    def __init__(self, args, preprocess, alpha=0.1):
        self.args = args
        self.preprocess = preprocess
        self.x_orig = None
        self.alpha = alpha

    def __call__(self, z):
        x = self.preprocess(z)
        x_ = self.alpha * self.x_orig + (1 - self.alpha) * x
        return x_


class ResizeOpt(Preprocessor):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        init_size = (params["input_size"], params["input_size"])
        final_size = (params["resize_size"], params["resize_size"])
        antialias = params["antialias"]
        interp = {
            "nearest": NEAREST,
            "bilinear": BILINEAR,
            "bicubic": BICUBIC,
        }[params["resize_interp"]]
        self.output_size = params["resize_size"]

        self.prep = transforms.Resize(
            final_size, interpolation=interp, antialias=antialias
        )

        self.inv_prep = transforms.Resize(
            init_size, interpolation=interp, antialias=antialias
        )
        self.inv_prep_proj = SimpleInterp(params, self.inv_prep, alpha=0.1)

        self.atk_prep = ApplySequence([self.inv_prep_proj, self.prep])
        self.prepare_atk_img = self.prep

    def set_x_orig(self, x):
        self.inv_prep_proj.x_orig = x
