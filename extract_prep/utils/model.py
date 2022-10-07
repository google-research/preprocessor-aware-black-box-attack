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

import torch.nn as nn


class PreprocessModel(nn.Module):
    def __init__(self, base_model, preprocess=None, normalize=None):
        super().__init__()
        self.base_model = base_model
        self.preprocess = preprocess
        self.normalize = normalize
        if normalize is not None:
            self.mean = nn.Parameter(
                normalize["mean"][None, :, None, None], requires_grad=False
            )
            self.std = nn.Parameter(
                normalize["std"][None, :, None, None], requires_grad=False
            )

    def forward(self, x):
        if self.preprocess is not None:
            x = x.clamp(0, 1)
            x = self.preprocess(x)
        x = x.clamp(0, 1)
        if self.normalize is not None:
            x = (x - self.mean) / self.std
        out = self.base_model(x)
        return out
