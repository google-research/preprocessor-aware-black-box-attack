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
