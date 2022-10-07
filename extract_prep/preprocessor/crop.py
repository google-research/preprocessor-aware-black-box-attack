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

from torchvision import transforms

from .base import BasePreprocess


class Crop(BasePreprocess):
    def __init__(self, params, input_size=None, **kwargs):
        super().__init__(params, input_size=input_size, **kwargs)
        # Assume square
        final_size = (params["crop_size"], params["crop_size"])
        pad_size = (input_size - params["crop_size"]) // 2
        assert pad_size > 0 and pad_size * 2 + params["crop_size"] == input_size
        self.output_size = params["crop_size"]
        self.pad_size = pad_size

        self.prep = transforms.CenterCrop(final_size)
        self.inv_prep = transforms.Pad(
            pad_size, fill=0.5, padding_mode="constant"
        )
        self.atk_prep = self.prep
        self.prepare_atk_img = self.prep
        self.atk_to_orig = self.inv_prep
        self.has_exact_project = True

    def project(self, z, x):
        x_proj = x.clone()
        size = z.shape[-1]
        pad = (x_proj.shape[-1] - size) // 2
        x_proj[:, :, pad : size + pad, pad : size + pad] = z
        return x_proj
