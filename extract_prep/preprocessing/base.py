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

from torch.nn import Module


def identity(x):
    return x


class BasePreprocess(Module):
    def __init__(self, params, input_size=None, **kwargs):
        super().__init__()
        self.prep = identity
        self.inv_prep = identity
        self.atk_prep = identity
        self.prepare_atk_img = identity
        self.output_size = input_size
        self.atk_to_orig = identity
        self.has_exact_project = False

    def get_prep(self):
        return self.prep, self.inv_prep, self.atk_prep, self.prepare_atk_img

    def set_x_orig(self, x):
        pass
