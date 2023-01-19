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
    HuggingfaceAPI,
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

    # NOTE: Specify your own set initial images and classification API here
    clf_pipeline: ClassifyAPI
    if clf_api == "local":
        prep_model: PreprocessModel
        prep_model, _ = setup_model(config, device=device)
        clf_pipeline = PyTorchModelAPI(prep_model)
        filenames = ["images/lena.png", "images/ILSVRC2012_val_00000293.jpg"]
    elif clf_api == "google":
        clf_pipeline = GoogleAPI()
        filenames = ["images/lena.png", "images/ILSVRC2012_val_00000293.jpg"]
    elif clf_api == "huggingface":
        clf_pipeline = HuggingfaceAPI(
            api_key=config["api_key"], model_url=config["model_url"]
        )
        filenames = ["images/lena.png", "images/ILSVRC2012_val_00000293.jpg"]
    elif clf_api == "imagga":
        clf_pipeline = ImaggaAPI(
            api_key=config["api_key"], api_secret=config["api_secret"]
        )
        filenames = ["images/lena.png", "tmp_nsfw.png"]
    elif clf_api == "sightengine":
        clf_pipeline = SightengineAPI(
            api_key=config["api_key"], api_secret=config["api_secret"]
        )
        filenames = ["images/lena.png", "tmp_nsfw.png"]
    else:
        raise NotImplementedError(
            f"{clf_api} classification API is not implemented!"
        )

    orig_size: tuple[int, int] = (config["orig_size"], config["orig_size"])
    dataset: list[np.ndarray] = [
        np.array(Image.open(fname).resize(orig_size))[..., :3]
        for fname in filenames
    ]
    dataset = np.stack(dataset)
    dataset = dataset.transpose((0, 3, 1, 2))

    # Initialize attack based on preprocessor to extract
    attack_fn = {
        "resize": FindResize,
        "crop": FindCrop,
    }[config["preprocess"]]
    attack = attack_fn(clf_pipeline, init_size=orig_size)
    dataset = attack.init(dataset)

    # Find unstable pair from dataset
    num_queries_total: int = 0
    find_unstable_pair = FindUnstablePair(clf_pipeline)
    unstable_pairs, num_queries = find_unstable_pair.find_unstable_pair(dataset)
    num_queries_total += num_queries

    # TODO: params
    # TODO: Have to guess the first preprocessor first unless they are exchandable
    num_trials: int = 20
    prep_params_guess = {
        # "output_size": [(224, 224), (256, 256), (512, 512)],
        "output_size": [(256, 256)],
        "interp": ["nearest", "bilinear", "bicubic"],
        "resize_lib": ["torch"],
    }

    # Create combinations of attack parameters and run attack
    keys, values = zip(*prep_params_guess.items())
    prep_params_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    candidate_params = []
    for i, prep_params in enumerate(prep_params_list):
        # Try one guessed parameter combination
        param_str = ", ".join(f"{k}={v}" for k, v in prep_params.items())
        logger.info(
            "Trying parameters %d/%d: %s",
            i + 1,
            len(prep_params_list),
            param_str,
        )
        num_succeeds: int = 0
        for _ in tqdm(range(num_trials)):
            is_successful, num_queries = attack.run(
                unstable_pairs,
                prep_params=prep_params,
                num_steps=config["num_extract_perturb_steps"],
            )
            num_succeeds += is_successful
            num_queries_total += num_queries

        logger.info(
            "# successes: %d/%d, # queries used so far: %d)",
            num_succeeds,
            num_trials,
            num_queries_total,
        )
        # If all trials are successful, keep parameter combination
        if num_succeeds == num_trials:
            candidate_params.append(prep_params)

    logger.info("Total number of queries: %d", num_queries_total)
    logger.info(
        "There are %d possible candidate(s) out of %d:\n%s",
        len(candidate_params),
        len(prep_params_list),
        candidate_params,
    )


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
