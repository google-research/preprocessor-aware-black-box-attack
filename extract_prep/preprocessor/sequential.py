#Copyright 2022 Google LLC
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

from .base import BasePreprocess
from .util import ApplySequence


class Sequential(BasePreprocess):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        from extract_prep.preprocessor import PREPROCESSORS

        # Get list of preprocessings from params
        preps = params["preprocess"].split("-")
        prep_list = []
        for prep in preps:
            input_size = (
                prep_list[-1].output_size if len(prep_list) > 0 else None
            )
            prep_list.append(PREPROCESSORS[prep](params, input_size=input_size))

        prep = [p.prep for p in prep_list]
        inv_prep = [p.inv_prep for p in prep_list][::-1]
        self.prep = ApplySequence(prep)
        self.inv_prep = ApplySequence(inv_prep)
        self.atk_prep = ApplySequence([*inv_prep, *prep])
        self.prepare_atk_img = ApplySequence(
            [p.prepare_atk_img for p in prep_list]
        )
        self.atk_to_orig = ApplySequence(
            [p.atk_to_orig for p in prep_list][::-1]
        )
        self.output_size = prep_list[-1].output_size
        self.prep_list = prep_list
