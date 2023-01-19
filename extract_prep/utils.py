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

import logging

import numpy as np
from PIL import Image
from tqdm import tqdm

from extract_prep.classification_api import ClassifyAPI

logger = logging.getLogger(__name__)


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
        "lanczos": Image.Resampling.LANCZOS,
        "area": Image.Resampling.BOX,
    }[interp]
    resized_imgs = []
    for img in imgs:
        img = img.transpose((1, 2, 0))
        img = Image.fromarray(img).resize(output_size, interp_mode)
        resized_imgs.append(np.array(img))
    resized_imgs = np.stack(resized_imgs).transpose((0, 3, 1, 2))
    return resized_imgs


def get_num_trials(
    config: dict[str, str | float | int],
    clf_pipeline: ClassifyAPI,
    unstable_pairs: np.ndarray,
    pval: float = 0.05,
    num_noises: int = 1000,
) -> int:
    """Get number of trials for the extraction attack."""
    if config["num_extract_trials"] is not None:
        logger.info("Using given num_trials: %d", config["num_extract_trials"])
        return config["num_extract_trials"]

    logger.info(
        "Given num_trials is None. Running a quick test to estimate "
        "num_trials that yield p-value of at most 0.05."
    )
    orig_labels = clf_pipeline(unstable_pairs)
    # Sample noise in unit L1 ball
    # https://mathoverflow.net/questions/9185/how-to-generate-random-points-in-ell-p-balls
    # noise = np.random.laplace(size=(num_noises,) + unstable_pairs[0].shape)
    # noise /=  np.abs(noise).sum(tuple(range(1, unstable_pairs.ndim)), keepdims=True)
    # noise *= config["num_extract_perturb_steps"]

    # randint num_noises times for each c, h, w
    
    perturbed = np.tile(unstable_pairs, (num_noises, 1, 1, 1))
    perturbed = perturbed.reshape((num_noises,) + unstable_pairs.shape)
    perturbed += noise[:, None]
    perturbed = np.clip(perturbed, 0, 255).astype(np.uint8)

    # Count the number of noises that change the labels
    num_diffs = 0
    for perturbed_i in tqdm(perturbed):
        labels = clf_pipeline(perturbed_i)
        if labels[0] != orig_labels[0] or labels[1] != orig_labels[1]:
            num_diffs += 1
    prob_diff = num_diffs / num_noises
    logger.info(
        "Prob of labels changing: %.4f (%d/%d)",
        prob_diff,
        num_diffs,
        num_noises,
    )

    # Compute num_trials to yield p-value of at most pval
    num_trials = np.ceil(np.log(pval) / np.log(1 - prob_diff))
    logger.info(
        "Estimated num_trials to yield p-value of at most 0.05: %d",
        num_trials,
    )
    return num_trials
