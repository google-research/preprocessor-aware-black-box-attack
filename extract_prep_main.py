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
from typing import Any

import numpy as np
import timm
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from attack_prep.preprocessor.util import (
    BICUBIC,
    BILINEAR,
    NEAREST,
    setup_preprocessor,
)
from attack_prep.utils.argparser import parse_args
from attack_prep.utils.model import PreprocessModel
from extract_prep.classification_api import ClassifyAPI, PyTorchModelAPI
from extract_prep.find_unstable_pair import FindUnstablePair


def setup_model(config: dict[str, Any], device: str = "cuda") -> nn.Module:
    """Set up plain PyTorch ImageNet classifier from timm."""
    normalize = dict(
        mean=torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32),
        std=torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32),
    )
    model = timm.create_model(config["model_name"], pretrained=True)
    preprocess = setup_preprocessor(config)
    prep = preprocess.get_prep()[0]

    # Initialize models with known and unknown preprocessing
    model = (
        PreprocessModel(model, preprocess=prep, normalize=normalize)
        .eval()
        .to(device)
    )
    model = nn.DataParallel(model).eval().to(device)
    return model


class FindBBox:
    def init(self, samples):
        raise

    def run(self, fn, bp):
        before = fn(bp)
        low = 0
        high = bp.shape[2] // 2
        while low + 1 < high:
            print("Search", low, high)
            mid = (high + low) // 2
            bp2 = np.copy(bp)
            bp2[:, :, mid] = 0
            after = fn(bp2)
            print("    ", after)
            if np.all(before == after):
                low = mid
            else:
                high = mid
        print("Done", mid)


class FindImageSize:
    def __init__(
        self,
        classifier_api: ClassifyAPI,
        orig_size: tuple[int, int] = (256, 256),
    ) -> None:
        self._classifier_api: ClassifyAPI = classifier_api
        self._orig_size: tuple[int, int] = orig_size

    def _pil_resize(
        self,
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
        samples = self._pil_resize(
            samples, output_size=self._orig_size, interp="nearest"
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
    ):
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


# def cheat(x):
#     return torch.stack(
#         [transform(Image.fromarray(np.array(y, dtype=np.uint8))) for y in x]
#     )


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

    # TODO: Get a classification pipeline
    prep_model: PreprocessModel = setup_model(config, device="cuda")
    clf_pipeline: ClassifyAPI = PyTorchModelAPI(prep_model)
    # g = get_clarifai
    # g(img1)
    # exit(0)

    # Find unstable pair for img1 and img2
    find_unstable_pair = FindUnstablePair(clf_pipeline)
    unstable_pairs: np.ndarray = find_unstable_pair.find_unstable_pair(dataset)

    attack = FindImageSize(clf_pipeline, orig_size=orig_size)
    dataset = attack.init(dataset)

    # bpoints = np.array(find_better_boundary_points(g, attack.init([img2, img1])))
    # attack.run(g, bpoints)

    # bpoints = np.array(find_better_boundary_points(g, attack.init([img2, img1])))
    # np.save("boundary_imagga2.npy", bpoints)
    # bpoints = np.array(np.load("boundary_imagga.npy"))

    prep_params = {
        "output_size": (224, 224),
        "interp": "nearest",
    }
    num_trials: int = 20
    num_succeeds: int = 0
    for _ in tqdm(range(num_trials)):
        is_successful = attack.run(
            unstable_pairs, prep_params=prep_params, num_steps=100
        )
        num_succeeds += is_successful
    print(f"{num_succeeds}/{num_trials}")

    exit(0)

    # exit(0)
    # return

    # find_bbox(g, bpoints)
    # return

    bp = np.array(bpoints)

    bpa = np.array(bp)
    bpb = np.array(bp)
    bpb[:, :, 0, :] = 0

    before = g(bpa)

    after = g(bpb)  #  + np.random.normal(0, .001, size=bpoints.shape))

    print("bp is", before)
    print("bp is", after)

    print("mean same", np.mean(before == after))
    return

    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[8:10, 8:10, 0] = 255

    Image.fromarray(img).save("/tmp/a.jpg", "JPEG", quality=50)
    r = Image.open("/tmp/a.jpg")
    print(np.array(r)[:18, :18, 0])
    print(np.array(r)[:, :, 0].mean(1))

    out = model(transform(Image.fromarray(img))[None])


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
