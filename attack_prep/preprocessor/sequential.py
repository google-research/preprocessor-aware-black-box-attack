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

"""Sequence of preprocessors."""

from __future__ import annotations

import copy
from typing import Any

from attack_prep.preprocessor.base import Preprocessor
from attack_prep.preprocessor.util import ApplySequence, setup_preprocessor


class Sequential(Preprocessor):
    """Sequentially apply a list of preprocessors."""

    def __init__(self, params: dict[str, Any], **kwargs) -> None:
        """Initialize sequence of preprocessors.

        Args:
            params: Config parameters.
        """
        super().__init__(params, **kwargs)
        # Get list of preprocessings from params
        preps = params["preprocess"].split("-")
        prep_list = []
        for prep in preps:
            temp_params = copy.deepcopy(params)
            input_size = (
                prep_list[-1].output_size if prep_list else params["orig_size"]
            )
            temp_params["preprocess"] = prep
            temp_params["orig_size"] = input_size
            prep_list.append(setup_preprocessor(temp_params))

        prep = [p.prep for p in prep_list]
        inv_prep = [p.inv_prep for p in prep_list][::-1]
        self.prep = ApplySequence(prep)
        self.inv_prep = ApplySequence(inv_prep)
        self.output_size = prep_list[-1].output_size
        self.prep_list = prep_list
