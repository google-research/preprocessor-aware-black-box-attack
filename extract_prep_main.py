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

import itertools
import logging
import os
import random
import sys

import numpy as np
import torch
from PIL import Image
from torch.backends import cudnn
from tqdm import tqdm

from attack_prep.utils.argparser import parse_args
from attack_prep.utils.model import PreprocessModel, setup_model
from extract_prep.classification_api import (
    ClassifyAPI,
    GoogleAPI,
    ImaggaAPI,
    PyTorchModelAPI,
    SightengineAPI,
)
from extract_prep.extractor import FindCrop
from extract_prep.find_unstable_pair import FindUnstablePair
from extract_prep.resize_extractor import FindResize

logger = logging.getLogger(__name__)


def _main() -> None:
    device: str = "cuda"
    clf_api: str = config["api"]

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])

    # Setting benchmark to True may result in non-deterministic results with
    # resizing.
    cudnn.benchmark = False
    # Setting deterministic must be set to True for neural-based preprocessor.
    # Othwerwise, the preprocessor itself may be non-deterministic.
    cudnn.deterministic = any(
        prep in config["preprocess"] for prep in ("neural", "sr")
    )

    orig_size: tuple[int, int] = (config["orig_size"], config["orig_size"])

    # NOTE: Specify your own set initial images here.
    filenames = ["images/lena.png", "tmp_nsfw.png"]
    dataset: list[np.ndarray] = [
        np.array(Image.open(fname).resize(orig_size))[..., :3]
        for fname in filenames
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
    clf_pipeline: ClassifyAPI
    if clf_api == "local":
        prep_model: PreprocessModel
        prep_model, _ = setup_model(config, device=device)
        clf_pipeline = PyTorchModelAPI(prep_model)
    elif clf_api == "google":
        clf_pipeline = GoogleAPI()
    elif clf_api == "imagga":
        clf_pipeline = ImaggaAPI()
    elif clf_api == "sightengine":
        clf_pipeline = SightengineAPI()
    else:
        raise NotImplementedError(
            f"{clf_api} classification API is not implemented!"
        )

    # import pdb
    # pdb.set_trace()
    # clf_pipeline(dataset[1])

    # Initialize attack based on preprocessor to extract
    attack_fn = {
        "resize": FindResize,
        "crop": FindCrop,
    }[config["preprocess"]]
    attack = attack_fn(clf_pipeline, init_size=orig_size)
    dataset = attack.init(dataset)

    # Find unstable pair from dataset
    find_unstable_pair = FindUnstablePair(clf_pipeline)
    unstable_pairs: np.ndarray = find_unstable_pair.find_unstable_pair(dataset)

    # TODO: params
    # TODO: Have to guess the first preprocessor first unless they are exchandable
    num_trials: int = 20
    prep_params_guess = {
        "output_size": [(224, 224), (256, 256), (512, 512)],
        "interp": ["nearest", "bilinear", "bicubic"],
    }

    # Create combinations of attack parameters and run attack
    keys, values = zip(*prep_params_guess.items())
    prep_params_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    for prep_params in prep_params_list:
        param_str = ", ".join(f"{k}={v}" for k, v in prep_params.items())
        logger.info("Trying parameters: %s", param_str)
        num_succeeds: int = 0
        for _ in tqdm(range(num_trials)):
            is_successful = attack.run(
                unstable_pairs, prep_params=prep_params, num_steps=100
            )
            num_succeeds += is_successful
        print(f"{num_succeeds}/{num_trials}")

# unknown variables:
# - compression
#   * jpeg
# - initial resize
#   * size of the resize (e.g., 256x256)
#   * mode for the resize (e.g., bilinear or nearest)
# - crop
#   * size of the crop
#   * center crop vs top rght crop vs top left crop


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        args.verbose = True
    os.makedirs("./results", exist_ok=True)
    config: dict[str, int | float | str] = vars(args)

    log_level: int = (
        logging.DEBUG if config["debug"] or config["verbose"] else logging.INFO
    )
    FORMAT_STR = "[%(asctime)s - %(name)s - %(levelname)s]: %(message)s"
    formatter = logging.Formatter(FORMAT_STR)
    logging.basicConfig(
        stream=sys.stdout,
        format=FORMAT_STR,
        level=log_level,
    )
    _main()
