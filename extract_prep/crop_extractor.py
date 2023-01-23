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

from extract_prep.extractor import FindPreprocessor

logger = logging.getLogger(__name__)


class FindCrop(FindPreprocessor):
    """Crop extractor."""

    def _apply_guess_prep(
        self,
        imgs: np.ndarray,
        crop_size: int = 10,
        border: str = "left",
    ) -> np.ndarray:
        assert isinstance(crop_size, int) and crop_size >= 0
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

    def _repeat_guess_and_check(
        self,
        unstable_pairs: np.ndarray,
        before: np.ndarray,
        crop_size: int = 10,
        border: str = "left",
        num_trials: int = 1,
    ) -> bool:
        for _ in range(num_trials):
            guess = self._apply_guess_prep(
                unstable_pairs, crop_size=crop_size, border=border
            )
            after = self._classifier_api(guess)
            self._num_queries += len(guess)
            if not np.all(before == after):
                return False
        return True

    def _binary_search(
        self,
        unstable_pairs: np.ndarray,
        before: np.ndarray,
        border: str = "left",
        num_trials: int = 1,
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
        logger.info("Binary search %s crop...", border)
        _, _, height, width = unstable_pairs.shape
        low = 0
        high = width // 2 if border in ("left", "right") else height // 2

        while low + 1 < high:
            logger.debug("low: %3d, high: %3d", low, high)
            mid = (high + low) // 2
            matched = self._repeat_guess_and_check(
                unstable_pairs,
                before,
                crop_size=mid,
                border=border,
                num_trials=num_trials,
            )
            # logger.debug("    %s", str(after))
            if matched:
                # If prediction does not change, increase cropped size
                low = mid
            else:
                high = mid
        logger.debug("Done. Found crop size of %3d", mid)
        return low

    def _run_binary_search(
        self,
        unstable_pairs: np.ndarray,
        before: np.ndarray,
        num_trials: int = 1,
    ) -> tuple[bool, int]:
        _, _, height, width = unstable_pairs.shape
        assert height == width

        # Binary search left crop
        left = self._binary_search(unstable_pairs, before, border="left")
        logger.info("Found left crop size of %3d", left)
        if left == 0:
            logger.warning("No cropping found. Returning...")
            return False, self._num_queries

        # Quickly check right crop for odd/even size
        found_right = False
        # Order of offsets is important here
        for offset in [0, 1, -1]:
            right = left + offset
            matched = self._repeat_guess_and_check(
                unstable_pairs,
                before,
                crop_size=right,
                border="right",
                num_trials=num_trials,
            )
            if matched:
                found_right = True
                logger.info("Found right crop size of %3d", right)
                break
        # If there's no match, binary search right crop
        if not found_right:
            logger.info("Right crop size is not found. Binary search again...")
            right = self._binary_search(unstable_pairs, before, border="right")

        # TODO: Assume square crop and left = top and right = bottom
        top, bottom = left, right

        # Apply the crops and check if the prediction changes
        # guess = unstable_pairs
        # for border, crop_size in zip(
        #     ("left", "right", "top", "bottom"), (left, right, top, bottom)
        # ):
        #     guess = self._apply_guess_prep(
        #         guess, crop_size=crop_size, border=border
        #     )
        #     prep_params[border] = crop_size
        # after = self._classifier_api(guess)
        # self._num_queries += len(guess)
        # is_matched = np.all(before == after)
        prep_params = {}
        for border, crop_size in zip(
            ("left", "right", "top", "bottom"), (left, right, top, bottom)
        ):
            prep_params[border] = crop_size
        return prep_params

    def _run_guess_and_check(
        self,
        unstable_pairs: np.ndarray,
        before: np.ndarray,
        prep_params: dict[str, int] | None = None,
    ) -> tuple[bool, int]:
        _, _, height, width = unstable_pairs.shape
        assert height == width
        guess = unstable_pairs
        for border, crop_size in prep_params.items():
            guess = self._apply_guess_prep(
                guess, crop_size=crop_size, border=border
            )
        after = self._classifier_api(guess)
        return np.all(before == after)

    def run(
        self,
        unstable_pairs: np.ndarray,
        before_labs: np.ndarray,
        prep_params: dict[str, int] | None = None,
        num_trials: int = 1,
        **kwargs,
    ) -> tuple[bool, int]:
        """Run the extractor.

        Args:
            unstable_pairs: _description_
            prep_params: If None, run binary search. Otherwise, run normal guess
                and check with given params. Defaults to None.
            num_trials: _description_. Defaults to 1.

        Returns:
            _description_
        """
        _ = kwargs  # Unused
        self._num_queries = 0
        if before_labs[0] == before_labs[1]:
            raise ValueError(
                "Predicted labels of unstable pair should be different!"
            )

        if prep_params is None:
            prep_params = self._run_binary_search(
                unstable_pairs, before_labs, num_trials=num_trials
            )
            return prep_params, self._num_queries

        is_matched = self._run_guess_and_check(
            unstable_pairs, before_labs, prep_params=prep_params
        )
        return is_matched, self._num_queries
