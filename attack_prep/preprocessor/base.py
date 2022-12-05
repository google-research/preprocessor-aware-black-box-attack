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

"""Base preprocessor module."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import Identity


class Preprocessor:
    """Base Preprocessor module."""

    def __init__(
        self,
        params: dict[str, str | int | float],
        input_size: int | None = None,
        **kwargs,
    ):
        """Initialize default preprocessor which is just an identity.

        Args:
            params: Params of preprocessors as a dictionary.
            input_size: Input image size (height, width). Defaults to None.
        """
        _ = params, kwargs  # Unused
        self.input_size: int | None = input_size
        self.output_size: int | None = input_size
        self.prep: nn.Module = Identity()
        self.inv_prep: nn.Module = Identity()
        self.has_exact_project: bool = False

    def get_prep(self) -> tuple[nn.Module, nn.Module]:
        """Return preprocessing function and its inverse."""
        return self.prep, self.inv_prep

    def set_x_orig(self, x_orig: torch.Tensor) -> None:
        """Keep original input in case some (inverse-)preprocessor needs it."""
