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

from abc import ABC


class Attack(ABC):
    def __init__(self, model, args, epsilon=None, input_size=224, **kwargs):
        self.model = model
        self.num_classes = args["num_classes"]
        self.epsilon = epsilon if epsilon is not None else args["epsilon"]
        self.input_size = input_size

    def run(self, imgs, labels):
        pass
