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

"""Find unstable pairs for extraction attack."""

from __future__ import annotations

from typing import Callable

import numpy as np


class FindUnstablePair:
    """Find unstable pair used for extraction attack."""

    def __init__(self, clf_pipeline) -> None:
        """Initialize FindUnstablePair.

        Args:
            clf_pipeline: Classification pipeline (preprocessors followed by
                a classifier).
        """
        self._clf_pipeline = clf_pipeline

    def _binary_search(
        self,
        left: np.ndarray,
        right: np.ndarray,
        choose_middle: Callable[
            [tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, bool]
        ],
    ) -> tuple[np.ndarray, np.ndarray]:
        left_label = self._clf_pipeline(left)
        while True:
            mid, done = choose_middle(left, right)
            if done:
                return left, right
            mid_label = self._clf_pipeline(mid)
            if left_label == mid_label:
                left = mid
            else:
                right = mid

    def _bisect_l0(
        self, left: np.ndarray, right: np.ndarray
    ) -> tuple[np.ndarray, bool]:
        """Find the "middle" between two images via L0-norm.

        Middle image has about a half of its pixels from left image and another
        half from right image.
        """
        mask = np.random.random_integers(0, 1, size=right.shape)
        mid = left * mask + right * (1 - mask)
        return mid, np.sum(left != right) == 1

    def _bisect_linf(
        self, left: np.ndarray, right: np.ndarray
    ) -> tuple[np.ndarray, bool]:
        """Find the "middle" between two images via Linf-norm.

        Each pixel of middle image is in the middle between the same pixel in
        left and right images.
        """
        left = left.astype(np.float32)
        right = right.astype(np.float32)
        mid = np.array((left + right) / 2, dtype=np.uint8)
        return mid, np.max(np.abs(left - right)) == 1

    def find_unstable_pair(self, dataset: np.ndarray) -> np.ndarray:
        """Find unstable pairs given a batch of samples."""
        if dataset.ndim != 4 and dataset.shape[1] == 3:
            raise ValueError(
                "dataset must be a batch of images (4D) with 2nd dim being the "
                f"color channels (3), but dataset.shape is {dataset.shape}!"
            )

        orig_outputs = self._clf_pipeline(dataset)
        img1: np.ndarray | None = None
        img2: np.ndarray | None = None
        if len(dataset) > 2:
            # Find a pair of images classified as different classes
            for i, x in enumerate(orig_outputs):
                for j, y in enumerate(orig_outputs[:i]):
                    if x != y:
                        img1 = dataset[i]
                        img2 = dataset[j]
                        break
        elif orig_outputs[0] != orig_outputs[1]:
            img1, img2 = dataset

        if img1 is None:
            raise ValueError(
                "All images in dataset are classified to the same class! "
                "We need at least two different classes."
            )

        print("Start L0 search...")
        img1, img2 = self._binary_search(img1, img2, self._bisect_l0)
        print("Start Linf search...")
        img1, img2 = self._binary_search(img1, img2, self._bisect_linf)
        l1, l2 = self._clf_pipeline([img1, img2])
        assert l1 != l2, (
            "Two samples in unstable pairs must be classified as two different "
            "classes."
        )
        print("Gap", np.sum(img1 - img2))
        return np.array((img1, img2))
