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

"""Common argparser."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Get a common argparser."""
    parser = argparse.ArgumentParser(
        description="Known-preprocessor Attack", add_help=True
    )
    parser.add_argument(
        "--name",
        default="",
        type=str,
        help="Custom experiment name for saving results.",
    )
    parser.add_argument(
        "--model-name",
        default="resnet18",
        type=str,
        help="Name of ImageNet classifier to load from timm.",
    )
    parser.add_argument(
        "--preprocess", required=True, type=str, help="Specify preprocessor."
    )
    parser.add_argument(
        "--attack",
        default="hopskipjump",
        type=str,
        help="Base attack algorithm to use.",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num-classes", default=1000, type=int)
    parser.add_argument(
        "--num-samples",
        default=16,
        type=int,
        help="Number of test samples to run attack on.",
    )
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument(
        "--workers",
        default=8,
        type=int,
        help="Number of workers for loading data.",
    )
    parser.add_argument("--data-dir", default="./data/", type=str)
    parser.add_argument(
        "--ord",
        default="2",
        type=str,
        help='Lp-norm of attack. Only L2 ("2") is supported at the moment.',
    )
    parser.add_argument("--epsilon", default=8 / 255, type=float)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--orig-size", default=224, type=int, help="Original image size."
    )
    parser.add_argument("--binary-search-steps", default=10, type=int)
    parser.add_argument("--lambda-search-steps", default=10, type=int)
    parser.add_argument(
        "--run-ukp-only",
        action="store_true",
        help="Run unknown-preprocessor attack only.",
    )
    parser.add_argument(
        "--run-kp-only",
        action="store_true",
        help="Run known-preprocessor attack only.",
    )
    parser.add_argument("--run-proj-only", default="", type=str)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save-adv", action="store_true")
    parser.add_argument("--mismatch-prep", default=None, type=str)
    parser.add_argument("--targeted", action="store_true")
    parser.add_argument(
        "--prep-grad-est",
        action="store_true",
        help=(
            "Apply preprocessing during attack gradient estimation (only "
            "works with HSJ, QEBA)."
        ),
    )
    parser.add_argument(
        "--prep-backprop",
        action="store_true",
        help=(
            "Backpropagate gradients through preprocessing during gradient "
            "approximation step (only works with HSJ, QEBA)."
        ),
    )
    parser.add_argument(
        "--prep-proj",
        action="store_true",
        help=("Project after each attack update (only works with HSJ, QEBA)."),
    )
    parser.add_argument(
        "--smart-noise",
        action="store_true",
        help="Enable Smart Noise Sampling (SNS) from Gao et al. [ICML 2022].",
    )
    # ============================= Preprocess ============================== #
    # Resize
    parser.add_argument("--resize-out-size", default=224, type=int)
    parser.add_argument("--resize-inv-size", default=None, type=int)
    parser.add_argument("--antialias", action="store_true")
    parser.add_argument("--resize-interp", default="nearest", type=str)
    # Quantize
    parser.add_argument("--quantize-num-bits", default=8, type=int)
    # Crop
    parser.add_argument("--crop-size", default=224, type=int)
    # JPEG
    parser.add_argument("--jpeg-quality", default=99.999, type=float)
    # Neural compression
    parser.add_argument("--neural-model", default=None, type=str)
    parser.add_argument("--neural-quality", default=6, type=int)
    # =============================== Attack ================================ #
    parser.add_argument("--max-iter", default=1000, type=int)
    # RayS
    parser.add_argument("--rays-num-queries", default=1000, type=int)
    # HopSkipJump
    parser.add_argument("--hsj-init-grad-steps", default=100, type=int)
    parser.add_argument("--hsj-max-grad-steps", default=200, type=int)
    parser.add_argument("--hsj-gamma", default=10, type=float)
    parser.add_argument(
        "--hsj-norm-rv",
        action="store_true",
        help=(
            "If True, normalize noise in gradient approximation step (same as "
            "the corrected implementation on Foolbox."
        ),
    )
    # Boundary Attack
    parser.add_argument("--boundary-step", default=1e-2, type=float)
    parser.add_argument("--boundary-orth-step", default=1e-2, type=float)
    # Square Attack
    parser.add_argument("--square-p", default=None, type=float)
    # Bandit Attack
    parser.add_argument("--bandit-fd-eta", default=0.01, type=float)
    parser.add_argument("--bandit-image-lr", default=0.5, type=float)
    # Sign-OPT Attack
    parser.add_argument("--signopt-grad-query", default=200, type=int)
    parser.add_argument("--signopt-grad-bs", default=100, type=int)
    parser.add_argument("--signopt-alpha", default=0.2, type=float)
    parser.add_argument("--signopt-beta", default=0.001, type=float)
    parser.add_argument("--signopt-momentum", default=0, type=float)
    parser.add_argument("--signopt-tgt-init-query", default=0, type=int)
    # Opt Attack
    parser.add_argument("--opt-alpha", default=0.2, type=float)
    parser.add_argument("--opt-beta", default=0.001, type=float)
    # QEBA Attack
    parser.add_argument("--qeba-subspace", default="resize4", type=str)
    parser.add_argument("--qeba-gamma", default=0.01, type=float)
    return parser.parse_args()
