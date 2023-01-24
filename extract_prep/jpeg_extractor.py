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

"""JPEG compression extractors."""

from __future__ import annotations

import logging

import numpy as np
import torch
from PIL import Image

from extract_prep.resize_extractor import FindResize

logger = logging.getLogger(__name__)


class FindJpeg(FindResize):
    """JPEG compression extractor."""

    def _apply_guess_prep(
        self,
        imgs: np.ndarray,
        quality: int = 100,
    ) -> np.ndarray:
        """Save images as JPEG with guessed params."""
        if not isinstance(quality, int) or quality < 0 or quality > 100:
            raise ValueError(f"Invalid JPEG quality: {quality}!")
        jpeg_imgs = []
        for img in imgs:
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            orig_img_shape = img.shape
            pil_img = Image.fromarray(np.array(img, dtype=np.uint8))
            tmp_file_name = "/tmp/img_jpeg.jpg"
            pil_img.save(tmp_file_name, quality=quality)
            img = np.array(Image.open(tmp_file_name))[..., :3]
            assert img.shape == orig_img_shape
            jpeg_imgs.append(img)
        return np.stack(jpeg_imgs, axis=0)
