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

"""Neural compression preprocessor."""

from __future__ import annotations

import torch
from compressai import zoo
from torch import nn

from attack_prep.preprocessor.base import Identity, Preprocessor


class Compress(nn.Module):
    def __init__(self, compress_net: nn.Module) -> None:
        super().__init__()
        self._compress_net: nn.Module = compress_net

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.clamp(0, 1)
        outputs = self._compress_net(inputs)["x_hat"]
        outputs.clamp_(0, 1)
        return outputs


class NeuralCompressor(Preprocessor):
    def __init__(self, params: dict[str, str | int | float], **kwargs) -> None:
        super().__init__(params, **kwargs)
        if "cheng2020" in params["neural_model"]:
            assert 1 <= params["neural_quality"] <= 6
        else:
            assert 1 <= params["neural_quality"] <= 8
        net_func = {
            "bmshj2018_factorized": zoo.bmshj2018_factorized,
            "bmshj2018_hyperprior": zoo.bmshj2018_hyperprior,
            "mbt2018_mean": zoo.mbt2018_mean,
            "mbt2018": zoo.mbt2018,
            "cheng2020_anchor": zoo.cheng2020_anchor,
            "cheng2020_attn": zoo.cheng2020_attn,
        }[params["neural_model"]]
        net = (
            net_func(quality=params["neural_quality"], pretrained=True)
            .eval()
            .to("cuda")
        )
        for param in net.parameters():
            param.requires_grad = False

        self.prep: nn.Module = Compress(net)
        self.inv_prep: nn.Module = Identity()
