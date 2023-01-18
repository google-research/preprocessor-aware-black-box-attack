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

"""Utility function for extration attack."""

from __future__ import annotations

import numpy as np
from PIL import Image


def pil_resize(
    imgs: np.ndarray,
    output_size: tuple[int, int] = (224, 224),
    interp: str = "bilinear",
) -> np.ndarray:
    """Resize images with PIL."""
    interp_mode = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
    }[interp]
    resized_imgs = []
    for img in imgs:
        img = img.transpose((1, 2, 0))
        img = Image.fromarray(img).resize(output_size, interp_mode)
        resized_imgs.append(np.array(img))
    resized_imgs = np.stack(resized_imgs).transpose((0, 3, 1, 2))
    return resized_imgs
