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
from torch.nn import Identity

identity = Identity()


class Preprocessor:
    """Base Preprocessor module."""

    def __init__(
        self,
        params: dict[str, str | int | float],
        input_size: tuple[int, int] | None = None,
        **kwargs,
    ):
        """Initialize default preprocessor which is just an identity.

        Args:
            params: Params of preprocessors as a dictionary.
            input_size: Input image size (height, width). Defaults to None.
        """
        _ = params, kwargs  # Unused
        self.output_size: tuple[int, int] | None = input_size
        self.prep = identity
        self.inv_prep = identity
        self.atk_prep = identity
        self.prepare_atk_img = identity
        self.atk_to_orig = identity
        self.has_exact_project: bool = False

    def get_prep(self):
        """Return all preprocessing functions.

        Returns:
            Tuple of preprocessing function, its inverse, its variant for using
            with preprocessor-aware attack, and its variant for preparing images
            to be attacked.
        """
        return self.prep, self.inv_prep, self.atk_prep, self.prepare_atk_img

    def set_x_orig(self, x_orig: torch.Tensor):
        """Keep original input in case some (inverse-)preprocessor needs it."""
