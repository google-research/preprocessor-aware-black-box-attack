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

"""Preprocessor extractors."""

from __future__ import annotations

import logging

import numpy as np

from extract_prep.classification_api import ClassifyAPI
from extract_prep.utils import pil_resize

logger = logging.getLogger(__name__)


class FindPreprocessor:
    """Base class for preprocessor extractors."""

    def __init__(
        self,
        classifier_api: ClassifyAPI,
        init_size: tuple[int, int] = (256, 256),
    ) -> None:
        """Initialize base preprocessor extractor.

        Args:
            classifier_api: Classification API.
            init_size: Initial image size. Defaults to (256, 256).
        """
        self._classifier_api: ClassifyAPI = classifier_api
        self._init_size: tuple[int, int] = init_size
        self._num_queries: int = 0

    def init(self, samples: np.ndarray) -> np.ndarray:
        """Initialize input samples for unstable pairs.

        Args:
            samples: Input samples used to create unstable pairs.

        Returns:
            Initialized samples.
        """
        # samples = skimage.transform.resize(samples, self._orig_size)
        samples = pil_resize(
            samples, output_size=self._init_size, interp="nearest"
        )
        # Clipped samples so they can be perturbed later
        return np.clip(samples, 30, 225)


class FindCrop(FindPreprocessor):
    """Crop extractor."""

    def _guess(
        self,
        imgs: np.ndarray,
        crop_size: int = 10,
        border: str = "left",
    ) -> np.ndarray:
        imgs = np.copy(imgs)
        if border in ("left", "right"):
            noise = np.random.randint(
                0, 256, size=imgs[:, :, :, :crop_size].shape
            )
            if border == "left":
                imgs[:, :, :, :crop_size] = noise
            else:
                imgs[:, :, :, -crop_size:] = noise
        else:
            noise = np.random.randint(
                0, 256, size=imgs[:, :, :crop_size, :].shape
            )
            if border == "top":
                imgs[:, :, :crop_size, :] = noise
            else:
                imgs[:, :, -crop_size:, :] = noise
        return imgs

    def _binary_search(
        self,
        unstable_pairs: np.ndarray,
        before: np.ndarray,
        border: str = "left",
    ) -> int:
        """Binary search cropped size on a given border.

        Args:
            unstable_pairs: Unstable pair.
            before: Original classification outcome.
            border: Which border to search ("left", "top", "right", "bottom").
                Defaults to "left".

        Returns:
            Number of pixels that are cropped from the given border.
        """
        print(f"Binary search {border} crop...")
        _, _, height, width = unstable_pairs.shape
        low = 0
        high = width // 2 if border in ("left", "right") else height // 2

        while low + 1 < high:
            print("Search", low, high)
            mid = (high + low) // 2
            bp2 = self._guess(unstable_pairs, crop_size=mid, border=border)
            after = self._classifier_api(bp2)
            print("    ", after)
            if np.all(before == after):
                # If prediction does not change, increase cropped size
                low = mid
            else:
                high = mid
        print("Done", mid)
        return low

    def run(
        self,
        unstable_pairs: np.ndarray,
        **kwargs,
    ) -> tuple[int, int]:
        del kwargs  # Unused
        _, _, height, width = unstable_pairs.shape
        assert height == width
        before = self._classifier_api(unstable_pairs)

        # Binary search left crop
        left = self._binary_search(unstable_pairs, before, border="left")
        if left == 0:
            return unstable_pairs.shape[-2:]

        # Quickly check right crop for odd/even size
        found_right = False
        for offset in [1, 0, -1]:
            right = left + offset
            guess = self._guess(unstable_pairs, crop_size=right, border="right")
            if np.all(before == self._classifier_api(guess)):
                found_right = True
                break
        # If there's no match, binary search right crop
        if not found_right:
            right = self._binary_search(unstable_pairs, before, border="right")

        # TODO: Assume square crop and left = top and right = bottom
        top, bottom = left, right

        return (height - top - bottom, width - left - right)
