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

"""Resize extractors."""

from __future__ import annotations

import logging
import random

import numpy as np
import torch
from torchvision import transforms

from attack_prep.preprocessor.util import BICUBIC, BILINEAR, NEAREST
from extract_prep.extractor import FindPreprocessor
from extract_prep.utils import pil_resize

logger = logging.getLogger(__name__)


def _torch_resize(
    imgs: np.ndarray,
    output_size: tuple[int, int] = (224, 224),
    interp: str = "bilinear",
) -> np.ndarray:
    interp_mode = {
        "nearest": NEAREST,
        "bilinear": BILINEAR,
        "bicubic": BICUBIC,
    }[interp]
    resize = transforms.Resize(
        output_size, interpolation=interp_mode, antialias=False
    )
    resized_imgs = resize(torch.from_numpy(imgs)).numpy()
    return resized_imgs


class FindResize(FindPreprocessor):
    """Resize extractor."""

    def _apply_guess_prep(
        self,
        imgs: np.ndarray,
        output_size: tuple[int, int] = (224, 224),
        interp: str = "bilinear",
        resize_lib: str = "pil",
    ) -> np.ndarray:
        """Resize images with guessed params."""
        # TODO: Implement other resizing ops
        if resize_lib == "pil":
            resized_imgs = pil_resize(
                imgs, output_size=output_size, interp=interp
            )
        elif resize_lib == "torch":
            resized_imgs = _torch_resize(
                imgs, output_size=output_size, interp=interp
            )
        else:
            raise NotImplementedError(f"Unknown resize lib: {resize_lib}!")
        return np.array(resized_imgs)

    def run(
        self,
        unstable_pairs: np.ndarray,
        prep_params: dict[str, int | float | str] | None = None,
        num_steps: int = 50,
    ) -> bool:
        """Run extraction attack.

        Args:
            unstable_pairs: Given unstable pairs.
            prep_params: Parameters for guessed preprocessor. Defaults to None.
            num_steps: Num steps to perturb unstable pairs. Defaults to 50.

        Raises:
            ValueError: Unstable pairs have the same label.

        Returns:
            # TODO
        """
        # 256x256 linear 18%
        # 256x256 bicubic 24%
        # 299x299 linear 20%
        # 299x299 bicubic 38%
        if prep_params is None:
            prep_params = {}
        before_labs = self._classifier_api(unstable_pairs)
        if before_labs[0] == before_labs[1]:
            raise ValueError(
                "Predicted labels of unstable pair should be different!"
            )
        orig_guess = self._apply_guess_prep(unstable_pairs, **prep_params)
        bpa = np.copy(unstable_pairs)

        for _ in range(num_steps):
            # Get a random pixel to perturb
            channel, row, col = [
                np.random.randint(s) for s in unstable_pairs.shape[1:]
            ]
            sign = random.choice(np.array([-1, 1], dtype=np.uint8))
            bpa[:, channel, row, col] += sign
            bpa = np.clip(bpa, 0, 255)
            if np.any(self._apply_guess_prep(bpa, **prep_params) != orig_guess):
                # Revert change if new image is NOT same as the original guess
                bpa[:, channel, row, col] -= sign

        assert np.max(bpa) <= 255 and np.min(bpa) >= 0
        logger.info(
            "Total diff: %d",
            np.sum(
                np.abs(bpa.astype(np.int32) - unstable_pairs.astype(np.int32))
            )
            / 2,
        )

        after_labs = self._classifier_api(bpa)
        return np.all(before_labs == after_labs)
