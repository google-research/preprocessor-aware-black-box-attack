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

"""Quantization preprocessor."""

from __future__ import annotations

import torch
from torch import nn

from attack_prep.preprocessor.base import Preprocessor, identity


class Quant(nn.Module):
    def __init__(self, num_bits: int) -> None:
        self.num_bits: int = num_bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x * (2**self.num_bits - 1)) / (
            2**self.num_bits - 1
        )


class Quantize(Preprocessor):
    def __init__(self, params: dict[str, str | int | float], **kwargs) -> None:
        super().__init__(params, **kwargs)
        self.prep = Quant(params["quantize_num_bits"])
        self.inv_prep = identity
        self.atk_prep = self.prep
        self.prepare_atk_img = identity
