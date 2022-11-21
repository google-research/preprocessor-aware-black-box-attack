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

"""Collections of all classification APIs."""

from __future__ import annotations

import abc
import io
import json
from typing import Any

import numpy as np
import requests
import torch
from google.cloud import vision
from PIL import Image

from attack_prep.utils.model import PreprocessModel

_TMP_IMG_PATH = "/tmp/img.png"
_TIMEOUT = 10


class ClassifyAPI:
    def __init__(self, tmp_img_path: str = _TMP_IMG_PATH, **kwargs) -> None:
        self._tmp_img_path = tmp_img_path

    @abc.abstractmethod
    def _run_one(self, img: np.ndarray) -> int | bool:
        pass

    def __call__(self, images: np.ndarray | list[np.ndarray]) -> np.ndarray:
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images]
        return np.array([self._run_one(x) for x in images])


def get_resnet(x, queries=[0]):
    if isinstance(x, list) or len(x.shape) == 4:
        o = model(cheat(x)).argmax(1).cpu().numpy()
        queries[0] += len(x)
        print("q", queries[0])
        # print(o)
        return o
    else:
        return get_resnet([x])[0]


def get_imagga(img):
    if isinstance(img, list) or len(img.shape) == 4:
        return np.array([get_imagga(y) for y in img])
    else:

        api_key = "acc_ef7c0ff4fd1d2e8"
        api_secret = "8f3dc6a875e2159da43cc40435fff5b1"
        image_path = "/tmp/img.png"

        Image.fromarray(np.array(img, dtype=np.uint8)).save(image_path)

        response = requests.post(
            "https://api.imagga.com/v2/categories/nsfw_beta",
            auth=(api_key, api_secret),
            files={"image": open(image_path, "rb")},
            timeout=_TIMEOUT,
        )
        print(response.json())
        for cat in response.json()["result"]["categories"]:
            if cat["name"]["en"] == "nsfw":
                return cat["confidence"] > 50
        return False


def get_sightengine(img):
    if isinstance(img, list) or len(img.shape) == 4:
        return np.array([get_sightengine(y) for y in img])
    else:
        params = {
            "models": "nudity",
            "api_user": "516789996",
            "api_secret": "imnX7kNtsY2SX4nTmuww",
        }
        Image.fromarray(np.array(img, dtype=np.uint8)).save("/tmp/img.png")
        files = {"media": open("/tmp/img.png", "rb")}
        r = requests.post(
            "https://api.sightengine.com/1.0/check.json",
            files=files,
            data=params,
            timeout=_TIMEOUT,
        )

        output = json.loads(r.text)
        print(output)
        return output["nudity"]["raw"] > 0.5


class GoogleAPI(ClassifyAPI):
    """Google Cloud Vision Safe Search API."""

    def _run_one(self, img: np.ndarray) -> int | bool:
        client = vision.ImageAnnotatorClient()
        Image.fromarray(np.array(img, dtype=np.uint8)).save(self._tmp_img_path)
        with io.open(self._tmp_img_path, "rb") as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        response = client.safe_search_detection(image=image)
        safe = response.safe_search_annotation
        return safe.adult == 1


class PyTorchModelAPI(ClassifyAPI):
    """Local PyTorch ImageNet classifier."""

    def __init__(
        self,
        prep_model: PreprocessModel | None = None,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._prep_model: PreprocessModel = prep_model
        self._device: str = device

    def _run_one(self, img: np.ndarray) -> int | bool:
        timg: torch.Tensor = torch.from_numpy(img).float()
        timg = timg.to(self._device)
        timg /= 255
        if timg.ndim == 3:
            timg.unsqueeze_(0)
        return self._prep_model(timg).argmax(1).cpu().numpy()
