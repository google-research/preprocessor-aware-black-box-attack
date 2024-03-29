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

import copy
import itertools
import logging
import os
import pickle
import random
import sys

import attack_prep.utils.backward_compat  # pylint: disable=unused-import

# pylint: disable=wrong-import-order
import huggingface_hub as hf_hub
import numpy as np
import requests
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
    ResponseError,
    SightengineAPI,
)
from extract_prep.crop_extractor import FindCrop
from extract_prep.find_unstable_pair import FindUnstablePair, UnstablePairError
from extract_prep.resize_extractor import FindResize
from extract_prep.jpeg_extractor import FindJpeg
from extract_prep.utils import get_num_trials

logger = logging.getLogger(__name__)


def _main(config: dict[str, str | int | float]) -> None:
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
        prep in config["preprocess"] for prep in ("neural", "sr", "denoise")
    )

    # NOTE: Specify your own set initial images and classification API here
    logger.info("Setting up %s classification API...", clf_api)
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
        # TODO
        # clf_pipeline = HuggingfaceAPI(
        #     api_key=config["api_key"], model_url=config["model_url"]
        # )
        jpeg_quality = None
        if config["preprocess"] == "jpeg":
            jpeg_quality = random.randint(50, 100)
            logger.info("Using JPEG quality %d", jpeg_quality)
        clf_pipeline = HuggingfaceAPI(
            api_key=config["api_key"],
            model_url=config["model_url"],
            jpeg_quality=jpeg_quality,
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
        "jpeg": FindJpeg,
    }[config["preprocess"]]
    attack = attack_fn(clf_pipeline, init_size=orig_size)
    dataset = attack.init(dataset)

    # Find unstable pair from dataset
    logger.info("Finding an unstable pair for this API...")
    num_queries_total: int = 0
    find_unstable_pair = FindUnstablePair(clf_pipeline)
    (
        unstable_pairs,
        unstable_labels,
        num_queries,
    ) = find_unstable_pair.find_unstable_pair(dataset)
    num_queries_total += num_queries

    if config["preprocess"] == "crop":
        logger.info("Extracting crop parameters...")
        prep_params_list = [None]
        num_trials_crop = 5
        # output_sizes = (
        #     [
        #         (224, 224),
        #         (256, 256),
        #         (299, 299),
        #         (384, 384),
        #         (512, 512),
        #         (248, 248),
        #         (288, 288),
        #     ],
        # )
        candidate_params, num_queries = attack.run(
            unstable_pairs,
            unstable_labels,
            num_trials=num_trials_crop,
        )
        num_queries_total += num_queries
        results = [
            {
                "params": candidate_params,
                "num_trials": num_trials_crop,
                "num_queries": num_queries,
            }
        ]
        logger.info("Total number of queries: %d", num_queries_total)
        logger.info("Found cropping params: %s", candidate_params)
        return results, candidate_params, num_queries_total

    num_trials: int = get_num_trials(
        config,
        clf_pipeline,
        unstable_pairs,
        pval=config["extract_pval"],
        num_noises=1000,
    )

    # TODO: params
    # TODO: Have to guess the first preprocessor first unless they are exchandable
    if config["preprocess"] == "resize":
        # Guess resize parameters
        output_sizes = np.arange(200, 800)
        np.random.shuffle(output_sizes)
        output_sizes = [(int(size), int(size)) for size in output_sizes]

        prep_params_guess = {
            # "output_size": [
            #     (224, 224),
            #     (256, 256),
            #     (299, 299),
            #     (384, 384),
            #     (512, 512),
            #     (248, 248),
            #     (288, 288),
            # ],
            "output_size": output_sizes,
            # "output_size": [(224, 224)],
            # "interp": ["bicubic", "bilinear", "nearest"],
            "interp": ["bilinear", "bicubic"],
            "resize_lib": ["pil"],
        }
    elif config["preprocess"] == "jpeg":
        prep_params_guess = {
            "quality": np.arange(50, 101, dtype=np.int32).tolist()
        }
    else:
        raise NotImplementedError(
            f"{config['preprocess']} preprocessor is not implemented for "
            "extration attack!"
        )

    # Create combinations of attack parameters and run attack
    keys, values = zip(*prep_params_guess.items())
    prep_params_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    results, candidate_params = [], []
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
        cur_num_queries: int = 0
        for _ in tqdm(range(num_trials)):
            is_successful, num_queries = attack.run(
                unstable_pairs,
                unstable_labels,
                prep_params=prep_params,
                num_steps=config["num_extract_perturb_steps"],
            )
            cur_num_queries += num_queries
            if not is_successful:
                logger.info(
                    "Guessed params are incorrect. Moving on to next guess..."
                )
                break
            num_succeeds += is_successful
        num_queries_total += cur_num_queries

        logger.info(
            "# successes: %d/%d, # queries: %d (total %d)",
            num_succeeds,
            num_trials,
            cur_num_queries,
            num_queries_total,
        )
        results.append(
            {
                "params": prep_params,
                "num_succeeds": num_succeeds,
                "num_trials": num_trials,
                "num_queries": cur_num_queries,
            }
        )
        if config["api"] == "huggingface":
            results[-1]["model_url"] = config["model_url"]

        # If all trials are successful, keep parameter combination
        if num_succeeds == num_trials:
            candidate_params.append(prep_params)
            logger.info("Candidate found! Stopping search.")
            break

    logger.info("Total number of queries: %d", num_queries_total)
    logger.info(
        "There are %d possible candidate(s) out of %d:\n%s",
        len(candidate_params),
        len(prep_params_list),
        candidate_params,
    )
    return results, candidate_params, num_queries_total


def _run_hf_exp(base_config: dict[str, int | float | str]) -> list[str]:
    logger.info("Running HuggingFace experiment...")
    hf_api = hf_hub.HfApi()
    model_args = hf_hub.ModelSearchArguments()

    filt = hf_hub.ModelFilter(
        task=model_args.pipeline_tag.ImageClassification,
        library=model_args.library.PyTorch,
    )
    models = hf_api.list_models(filter=filt)
    # hf_hub.DatasetSearchArguments() is buggy so we go with searching
    # "imagenet" in tags instead
    models = filter(lambda m: any("imagenet" in t for t in m.tags), models)
    models = list(models)
    num_total: int = base_config["num_hf_models"]
    model_ids = random.sample([m.modelId for m in models], len(models))

    num_finished: int = 0
    all_results = []
    for model_id in model_ids:
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        logger.info(
            "[%3d/%3d]: %s (%s)", num_finished + 1, num_total, model_id, url
        )
        config = copy.deepcopy(base_config)
        config["model_url"] = url
        config["seed"] = config["seed"] + 1

        try:
            results, candidates, num_queries = _main(config)
        except (
            ResponseError,
            UnstablePairError,
            requests.exceptions.ReadTimeout,
        ) as err:
            logger.warning(err)
            logger.info("Skipping...")
            continue

        num_finished += 1
        all_results.append(
            {
                "model_id": model_id,
                "url": url,
                "results": results,
                "candidates": candidates,
                "num_queries": num_queries,
            }
        )
        logger.info("=" * 20)
        if num_finished >= num_total:
            break

    logger.info("Finished %d/%d models.", num_finished, num_total)
    result_path = "./results/hf_results.pkl"
    with open(result_path, "wb") as file:
        pickle.dump(all_results, file)
    logger.info("Saved results to %s", result_path)


# unknown variables:
# - compression
#   * jpeg
# - initial resize
#   * size of the resize (e.g., 256x256) 224, 256, 299, 384, 512
#   * mode for the resize (e.g., bilinear or nearest)
# - crop
#   * size of the crop
#   * center crop vs top rght crop vs top left crop

# Crop by % could be swappable with resize


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        args.verbose = True
    os.makedirs("./results", exist_ok=True)
    main_config: dict[str, int | float | str] = vars(args)

    log_level: int = (
        logging.DEBUG
        if main_config["debug"] or main_config["verbose"]
        else logging.INFO
    )
    FORMAT_STR = "[%(asctime)s - %(name)s - %(levelname)s]: %(message)s"
    formatter = logging.Formatter(FORMAT_STR)
    logging.basicConfig(
        stream=sys.stdout,
        format=FORMAT_STR,
        level=log_level,
    )
    if main_config["run_hf_exp"]:
        main_config["api"] = "huggingface"
        _run_hf_exp(main_config)
    else:
        _main(main_config)
