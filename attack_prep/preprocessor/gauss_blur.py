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

"""Gaussian blur preprocessor."""

from __future__ import annotations

from torch import nn
from torchvision import transforms

from attack_prep.preprocessor.base import Identity, Preprocessor


class GaussianBlur(Preprocessor):
    """Gaussian blur preprocessor."""

    def __init__(
        self, params: dict[str, str | int | float], **kwargs,
    ) -> None:
        """Initialize Gaussian blur.

        Args:
            params: Cropping parameters.
        """
        super().__init__(params, **kwargs)
        self.prep = transforms.GaussianBlur(
            params["blur_kernel_size"], sigma=params["blur_sigma"]
        )
        self.inv_prep: nn.Module = Identity()
