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

"""Utility functions for all preprocessors."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision import transforms

from attack_prep.preprocessor.base import Preprocessor

NEAREST = transforms.InterpolationMode.NEAREST
BILINEAR = transforms.InterpolationMode.BILINEAR
BICUBIC = transforms.InterpolationMode.BICUBIC


def setup_preprocessor(config: dict[str, Any]) -> Preprocessor:
    """Create preprocessor given config."""
    # Dirty import here to avoid circular imports
    # pylint: disable=import-outside-toplevel
    from attack_prep.preprocessor.crop import Crop
    from attack_prep.preprocessor.jpeg import JPEG
    from attack_prep.preprocessor.neural import Neural
    from attack_prep.preprocessor.quantize import Quantize
    from attack_prep.preprocessor.resize import Resize
    from attack_prep.preprocessor.sequential import Sequential

    if "-" in config["preprocess"]:
        preprocessor_fn = Sequential
    else:
        preprocessor_fn = {
            "identity": Preprocessor,
            "quantize": Quantize,
            "resize": Resize,
            "crop": Crop,
            "jpeg": JPEG,
            "neural": Neural,
        }[config["preprocess"]]

    preprocess = preprocessor_fn(config, input_size=config["orig_size"])
    return preprocess


class ApplySequence(nn.Module):
    """Apply multiple preprocessors sequentially."""

    def __init__(self, preprocesses: list[nn.Module]) -> None:
        """Initialize ApplySequence.

        Args:
            preprocesses: List of preprocessors to apply in order.
        """
        super().__init__()
        self.preprocesses: list[nn.Module] = preprocesses

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Preprocess."""
        for prep in self.preprocesses:
            inputs = prep(inputs)
        return inputs


class RgbToGrayscale(nn.Module):
    """Convert RGB image to grayscale."""

    def __init__(self) -> None:
        """Initialize RgbToGrayscale."""
        super().__init__()
        weight = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)[
            None, :, None, None
        ]
        self.register_buffer("_weight", weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Convert RGB image to grayscale and expands back to 3 channels."""
        return (
            (inputs * self._weight).sum(1, keepdim=True).expand(-1, 3, -1, -1)
        )
