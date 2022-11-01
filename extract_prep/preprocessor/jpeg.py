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

from extract_prep.preprocessor.base import Preprocessor, _identity
from extract_prep.preprocessor.diffjpeg import DiffJPEG


class JPEG(Preprocessor):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        quality = params["jpeg_quality"]
        self.prep = DiffJPEG(differentiable=True, quality=quality).cuda()
        self.inv_prep = _identity
        self.atk_prep = self.prep
        self.prepare_atk_img = _identity
