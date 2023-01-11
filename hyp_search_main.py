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

"""Run attack by also sweeping a set of pre-defined hyperparameters."""

import os
from functools import partial

from preprocess_attack_main import parse_args, run_one_setting

print = partial(print, flush=True)  # pylint: disable=redefined-builtin

if __name__ == "__main__":

    args = parse_args()
    os.makedirs("./results", exist_ok=True)
    if args.debug:
        args.verbose = True

    hyp_to_search = {
        "boundary": {
            "hyp": ["boundary_step", "boundary_orth_step"],
            "val": [[0.1, 0.01, 0.001], [0.1, 0.001]],
        },
        "signopt": {
            "hyp": ["signopt_alpha", "signopt_beta"],
            "val": [[2, 0.2, 0.02], [0.1, 0.01, 0.0001]],
        },
        "hsj": {
            "hyp": ["hsj_gamma"],
            "val": [[1e3, 1e4, 1e5, 1e6]],
            # "val": [[1e2, 1e3, 1e4, 1e5]],
        },
        "qeba": {
            "hyp": ["qeba_subspace"],
            "val": [
                [
                    # "naive",
                    "resize2",
                    "resize4",
                    "resize8",
                    "resize16",
                    # "resize32",
                ]
            ],
        },
    }

    if args.smart_noise:
        hyp_to_search["qeba"]["val"] = [["naive"]]

    params = hyp_to_search[args.attack]
    for i, hyp in enumerate(params["hyp"]):
        vals = params["val"][i]
        for val in vals:
            config = vars(args)
            config[hyp] = val
            run_one_setting(config)
