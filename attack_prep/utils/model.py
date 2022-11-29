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

"""Model utility functions."""

from __future__ import annotations

from typing import Any

import timm
import torch
from torch import nn

from attack_prep.preprocessor.base import Preprocessor
from attack_prep.preprocessor.util import setup_preprocessor


class PreprocessModel(nn.Module):
    """Classification pipeline with preprocessors."""

    def __init__(
        self,
        base_model: nn.Module,
        preprocess: nn.Module | None = None,
        normalize: dict[str, tuple[float, float, float]] | None = None,
    ) -> None:
        """Initialize PreprocessorModel.

        Args:
            base_model: Base plain classifier.
            preprocess: Preprocessor module. Defaults to None.
            normalize: Mean and standard deviation input normalization.
                Defaults to None.
        """
        super().__init__()
        self.base_model: nn.Module = base_model
        self.preprocess: nn.Module | None = preprocess
        self.normalize: dict[str, tuple[float, float, float]] | None = normalize
        if normalize is not None:
            self.mean = nn.Parameter(
                normalize["mean"][None, :, None, None], requires_grad=False
            )
            self.std = nn.Parameter(
                normalize["std"][None, :, None, None], requires_grad=False
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply preprocessing and then model."""
        if self.preprocess is not None:
            inputs = inputs.clamp(0, 1)
            inputs = self.preprocess(inputs)
        inputs = inputs.clamp(0, 1)
        if self.normalize is not None:
            inputs = (inputs - self.mean) / self.std
        out = self.base_model(inputs)
        return out


def setup_model(
    config: dict[str, Any],
    device: str = "cuda",
    # known_prep: bool = False,
) -> tuple[nn.Module, Preprocessor]:
    """Set up plain PyTorch ImageNet classifier from timm."""
    normalize = dict(
        mean=torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32),
        std=torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32),
    )
    model: nn.Module = timm.create_model(config["model_name"], pretrained=True)
    preprocess: Preprocessor = setup_preprocessor(config)
    prep, _ = preprocess.get_prep()

    # Initialize models with known and unknown preprocessing
    model: PreprocessModel = (
        PreprocessModel(model, preprocess=prep, normalize=normalize)
        .eval()
        .to(device)
    )
    return model, preprocess
