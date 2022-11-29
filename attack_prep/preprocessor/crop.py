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

"""Cropping preprocessor."""

from __future__ import annotations

import torch
from torchvision import transforms

from attack_prep.preprocessor.base import Preprocessor


class Crop(Preprocessor):
    """Cropping preprocessor."""

    def __init__(
        self,
        params: dict[str, str | int | float],
        input_size: int = 256,
        **kwargs,
    ) -> None:
        """Initialize Crop.

        Args:
            params: Cropping parameters.
            input_size: Input image width or height. Defaults to 256.
        """
        super().__init__(params, input_size=input_size, **kwargs)
        # Assume square
        final_size = (params["crop_size"], params["crop_size"])
        pad_size0 = (input_size - params["crop_size"]) // 2
        # For odd crop size, torch crops left and top one more pixel than right
        # and bottom
        pad_size1 = input_size - params["crop_size"] - pad_size0
        assert pad_size1 > 0, f"Pad size must be positive but is {pad_size0}!"

        self.output_size = params["crop_size"]
        self.prep = transforms.CenterCrop(final_size)
        # left, top, right, and bottom
        self.inv_prep = transforms.Pad(
            [pad_size1, pad_size1, pad_size0, pad_size0],
            fill=0.5,
            padding_mode="constant",
        )
        self.atk_prep = self.prep
        self.prepare_atk_img = self.prep
        self.atk_to_orig = self.inv_prep
        self.has_exact_project = True

    def project(
        self, z_adv: torch.Tensor, x_orig: torch.Tensor
    ) -> torch.Tensor:
        """Find projection of x onto z.

        This is the closed-form recovery phase for cropping. x represents the
        original image, and z is adversarial example in the processed space.

        Args:
            z: Original image.
            x: Adversarial example in processed space.

        Returns:
            Projection of x on z.
        """
        x_proj = x_orig.clone()
        size = z_adv.shape[-1]
        pad = (x_proj.shape[-1] - size) // 2
        x_proj[:, :, pad : size + pad, pad : size + pad] = z_adv
        return x_proj
