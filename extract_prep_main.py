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

"""Extract preprocessors from an ML pipeline."""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from attack_prep.preprocessor.util import BICUBIC, BILINEAR, NEAREST
from attack_prep.utils.argparser import parse_args
from attack_prep.utils.model import PreprocessModel, setup_model
from extract_prep.classification_api import (
    ClassifyAPI,
    GoogleAPI,
    PyTorchModelAPI,
)
from extract_prep.find_unstable_pair import FindUnstablePair


def _pil_resize(
    imgs: np.ndarray,
    output_size: tuple[int, int] = (224, 224),
    interp: str = "bilinear",
) -> np.ndarray:
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


class FindPreprocessor:
    def __init__(
        self,
        classifier_api: ClassifyAPI,
        init_size: tuple[int, int] = (256, 256),
    ) -> None:
        self._classifier_api: ClassifyAPI = classifier_api
        self._init_size: tuple[int, int] = init_size


class FindResize(FindPreprocessor):
    def _torch_resize(
        self,
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

    def init(self, samples: np.ndarray) -> np.ndarray:
        # samples = skimage.transform.resize(samples, self._orig_size)
        samples = _pil_resize(
            samples, output_size=self._init_size, interp="nearest"
        )
        # FIXME: why is clipping needed?
        # return np.clip(samples, 30, 225)
        return samples

    def _guess(
        self,
        imgs: np.ndarray,
        output_size: tuple[int, int] = (224, 224),
        interp: str = "bilinear",
    ) -> np.ndarray:
        """Resize images with guessed params."""
        # TODO: Implement other resizing ops
        # resized_imgs = self._pil_resize(
        #     imgs, output_size=output_size, interp=interp
        # )
        resized_imgs = self._torch_resize(
            imgs, output_size=output_size, interp=interp
        )
        return np.array(resized_imgs)

    def run(
        self,
        unstable_pairs: np.ndarray,
        prep_params: dict[str, int | float | str] | None = None,
        num_steps: int = 50,
    ) -> bool:
        # 256x256 linear 18%
        # 256x256 bicubic 24%
        # 299x299 linear 20%
        # 299x299 bicubic 38%
        before_labs = self._classifier_api(unstable_pairs)
        if before_labs[0] == before_labs[1]:
            raise ValueError(
                "Predicted labels of unstable pair should be different!"
            )
        orig_guess = self._guess(unstable_pairs, **prep_params)
        bpa = np.copy(unstable_pairs)

        for _ in range(num_steps):
            # Get random pixel to perturb
            channel, row, col = [
                np.random.randint(s) for s in unstable_pairs.shape[1:]
            ]
            sign = random.choice(np.array([-1, 1], dtype=np.uint8))
            bpa[:, channel, row, col] += sign
            bpa = np.clip(bpa, 0, 255)
            if np.any(self._guess(bpa, **prep_params) != orig_guess):
                # Flip sign if same as the original
                bpa[:, channel, row, col] -= sign

        assert np.max(bpa) <= 255 and np.min(bpa) >= 0
        print(
            "Total diff",
            np.sum(
                np.abs(bpa.astype(np.int32) - unstable_pairs.astype(np.int32))
            )
            / 2,
        )

        after_labs = self._classifier_api(bpa)
        return np.all(before_labs == after_labs)


class FindCrop(FindPreprocessor):
    def init(self, samples: np.ndarray) -> np.ndarray:
        samples = _pil_resize(
            samples, output_size=self._init_size, interp="nearest"
        )
        return samples

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


def _main(config: dict[str, int | float | str]) -> None:

    orig_size: tuple[int, int] = (config["orig_size"], config["orig_size"])

    # TODO: Use other images from test set
    filenames = ["images/lena.png", "images/ILSVRC2012_val_00000293.jpg"]
    dataset: list[np.ndarray] = [
        np.array(Image.open(fname).resize(orig_size)) for fname in filenames
    ]
    dataset = np.stack(dataset)
    dataset = dataset.transpose((0, 3, 1, 2))

    assert isinstance(
        dataset, np.ndarray
    ), f"dataset must be a NumPy array, but it is {type(dataset)}!"
    assert dataset.ndim == 4 and dataset.shape[1] == 3, (
        "dataset must have shape [batch, channel, height, width], but it has "
        f"shape {dataset.shape}!"
    )

    # TODO: Get a classification pipeline\
    if config["api"] == "local":
        prep_model: PreprocessModel = setup_model(config, device="cuda")
        clf_pipeline: ClassifyAPI = PyTorchModelAPI(prep_model)
    elif config["api"] == "google":
        clf_pipeline: ClassifyAPI = GoogleAPI()
    else:
        raise NotImplementedError(
            f"{config['api']} classification API is not implemented!"
        )

    # attack = FindResize(clf_pipeline, init_size=orig_size)
    attack = FindCrop(clf_pipeline, init_size=orig_size)
    dataset = attack.init(dataset)

    # Find unstable pair from dataset
    find_unstable_pair = FindUnstablePair(clf_pipeline)
    unstable_pairs: np.ndarray = find_unstable_pair.find_unstable_pair(dataset)

    # prep_params = {
    #     "output_size": (224, 224),
    #     "interp": "nearest",
    # }
    # num_trials: int = 20
    # num_succeeds: int = 0
    # for _ in tqdm(range(num_trials)):
    #     is_successful = attack.run(
    #         unstable_pairs, prep_params=prep_params, num_steps=100
    #     )
    #     num_succeeds += is_successful
    # print(f"{num_succeeds}/{num_trials}")

    print(attack.run(unstable_pairs))


"""
unknown variables:
- compression
  * jpeg
- initial resize
  * size of the resize (e.g., 256x256)
  * mode for the resize (e.g., bilinear or nearest)
- crop
  * size of the crop
  * center crop vs top rght crop vs top left crop
"""


if __name__ == "__main__":
    args = parse_args()
    os.makedirs("./results", exist_ok=True)
    if args.debug:
        args.verbose = True
    _main(vars(args))
