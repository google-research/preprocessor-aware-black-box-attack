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

import numpy as np
import requests
import torch
from google.cloud import vision
from PIL import Image

from attack_prep.utils.model import PreprocessModel

_TMP_IMG_PATH = "/tmp/img.png"
_TIMEOUT = 10


class ClassifyAPI:
    """Base ClassifyAPI wraps any classification pipeline."""

    def __init__(
        self,
        tmp_img_path: str = _TMP_IMG_PATH,
        timeout: int = _TIMEOUT,
        **kwargs,
    ) -> None:
        self._tmp_img_path: str = tmp_img_path
        self._timeout: int = timeout

    @abc.abstractmethod
    def _run_one(self, img: np.ndarray) -> int | bool:
        pass

    def __call__(self, images: np.ndarray | list[np.ndarray]) -> np.ndarray:
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images]
        return np.array([self._run_one(x) for x in images])


class ImaggaAPI(ClassifyAPI):
    """Imagga NSFW (beta) API."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._api_key = "acc_ef7c0ff4fd1d2e8"
        self._api_secret = "8f3dc6a875e2159da43cc40435fff5b1"

    def _run_one(self, img: np.ndarray | torch.Tensor) -> int | bool:
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        Image.fromarray(np.array(img, dtype=np.uint8)).save(self._tmp_img_path)
        response = requests.post(
            "https://api.imagga.com/v2/categories/nsfw_beta",
            auth=(self._api_key, self._api_secret),
            files={"image": open(self._tmp_img_path, "rb")},
            timeout=self._timeout,
        )
        response = response.json()
        assert (
            response["status"]["type"] == "success"
        ), "Failed response from Imagga API!"

        max_cat, max_conf = None, -1
        for cat in response["result"]["categories"]:
            if cat["confidence"] > max_conf:
                max_conf = cat["confidence"]
                max_cat = cat["name"]["en"]
        return max_cat == "nsfw"


class SightengineAPI(ClassifyAPI):
    """Sightengine nudity API."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._api_user = "516789996"
        self._api_secret = "imnX7kNtsY2SX4nTmuww"

    def _run_one(self, img: np.ndarray) -> int | bool:
        params = {
            "models": "nudity-2.0",
            "api_user": self._api_user,
            "api_secret": self._api_secret,
        }
        Image.fromarray(np.array(img, dtype=np.uint8)).save(self._tmp_img_path)
        files = {"media": open(self._tmp_img_path, "rb")}
        response = requests.post(
            "https://api.sightengine.com/1.0/check.json",
            files=files,
            data=params,
            timeout=self._timeout,
        )
        output = json.loads(response.text)
        print(output)
        import pdb

        pdb.set_trace()
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
