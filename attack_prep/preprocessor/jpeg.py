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

from __future__ import annotations

from torch import nn

from attack_prep.preprocessor.base import Identity, Preprocessor
from attack_prep.preprocessor.diffjpeg.jpeg import DiffJPEG


class JPEG(Preprocessor):
    def __init__(self, params: dict[str, str | int | float], **kwargs) -> None:
        super().__init__(params, **kwargs)
        quality = params["jpeg_quality"]
        self.prep: nn.Module = DiffJPEG(
            differentiable=True, quality=quality
        ).cuda()
        self.inv_prep: nn.Module = Identity()
