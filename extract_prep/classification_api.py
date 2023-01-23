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
import logging
import time
import uuid

import numpy as np
import requests
import torch
from google.cloud import vision
from PIL import Image

from attack_prep.utils.model import PreprocessModel

_TIMEOUT = 10
logger = logging.getLogger(__name__)


class ResponseError(Exception):
    """Raised when online API returns an error response."""


class ClassifyAPI:
    """Base ClassifyAPI wraps any classification pipeline."""

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        tmp_img_path: str | None = None,
        timeout: int = _TIMEOUT,
        **kwargs,
    ) -> None:
        """Initialize ClassifyAPI.

        Args:
            api_key: API key for online API.
            api_secret: API secret for online API.
            tmp_img_path: Path to temporarily save image before sending it via
                POST request. Defaults to _TMP_IMG_PATH.
            timeout: Timeout for online API. Defaults to _TIMEOUT.
        """
        _ = kwargs  # Unused
        self._api_key: str | None = api_key
        self._api_secret: str | None = api_secret
        self._timeout: int = timeout
        # Generate a unique ID for each tmp file
        if tmp_img_path is None:
            uid = str(uuid.uuid4())
            tmp_img_path = f"/tmp/img_{uid}.png"
        self._tmp_img_path: str = tmp_img_path

    @abc.abstractmethod
    def _run_one(self, img: np.ndarray) -> int | bool:
        pass

    def __call__(self, images: np.ndarray | list[np.ndarray]) -> np.ndarray:
        """Run the classification pipeline on a batch of images."""
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images]
        return np.array([self._run_one(x) for x in images])


class ImaggaAPI(ClassifyAPI):
    """Imagga NSFW (beta) API."""

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
        if response["status"]["type"] != "success":
            raise ResponseError(f"Failed response from Imagga API!\n{response}")

        max_cat, max_conf = None, -1
        for cat in response["result"]["categories"]:
            if cat["confidence"] > max_conf:
                max_conf = cat["confidence"]
                max_cat = cat["name"]["en"]
        return max_cat == "nsfw"


class SightengineAPI(ClassifyAPI):
    """Sightengine nudity API."""

    def _run_one(self, img: np.ndarray) -> int | bool:
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        Image.fromarray(np.array(img, dtype=np.uint8)).save(self._tmp_img_path)

        params = {
            "models": "nudity-2.0",
            "api_user": self._api_key,
            "api_secret": self._api_secret,
        }
        files = {"media": open(self._tmp_img_path, "rb")}

        response = requests.post(
            "https://api.sightengine.com/1.0/check.json",
            files=files,
            data=params,
            timeout=self._timeout,
        )
        output = json.loads(response.text)
        if output["status"] != "success":
            raise ResponseError(
                f"Failed response from Sightengine API!\n{output}"
            )

        # Find the most confident category
        max_cat, max_conf = None, -1
        for cat, conf in output["nudity"].items():
            if not isinstance(conf, float):
                continue
            if conf > max_conf:
                max_conf = conf
                max_cat = cat
        return max_cat != "none"


class GoogleAPI(ClassifyAPI):
    """Google Cloud Vision Safe Search API."""

    def _run_one(self, img: np.ndarray) -> int | bool:
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        Image.fromarray(np.array(img, dtype=np.uint8)).save(self._tmp_img_path)

        with io.open(self._tmp_img_path, "rb") as image_file:
            content = image_file.read()

        client = vision.ImageAnnotatorClient()

        # pylint: disable=no-member
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
        """Initialize the PyTorch model API.

        Args:
            prep_model: Model with preprocessor. Defaults to None.
            device: Device of model. Defaults to "cuda".
        """
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


class HuggingfaceAPI(ClassifyAPI):
    """Hugging Face inference API."""

    def __init__(
        self,
        model_url: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize the Hugging Face API.

        Args:
            model_url: URL of model to use inference from. Defaults to None.
        """
        super().__init__(**kwargs)
        self._model_url: str = model_url

    def _run_one(self, img: np.ndarray) -> str:
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        Image.fromarray(np.array(img, dtype=np.uint8)).save(self._tmp_img_path)

        with open(self._tmp_img_path, "rb") as file:
            data = file.read()

        def _post_request():
            response = requests.request(
                "POST",
                self._model_url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                data=data,
                timeout=self._timeout,
            )
            output = json.loads(response.content.decode("utf-8"))
            return output

        num_tries = 0
        while True:
            output = _post_request()
            if "error" in output and "is currently loading" in output["error"]:
                # Wait 10% longer than the estimated time
                wait_time = int(output["estimated_time"] * 1.1)
                logger.info(
                    "%s, waiting %d seconds", output["error"], wait_time
                )
                if wait_time > 60 or num_tries >= 3:
                    raise ResponseError(
                        f"Model took too long to load ({num_tries} retries)!"
                    )
                time.sleep(wait_time)
                num_tries += 1
            elif isinstance(output, list) and "label" in output[0]:
                break
            else:
                raise ResponseError(
                    f"Failed response from Hugging Face API!\n{output}"
                )

        return output[0]["label"]
