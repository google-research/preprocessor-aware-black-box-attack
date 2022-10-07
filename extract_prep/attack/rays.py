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

import numpy as np
from RayS.general_torch_model import GeneralTorchModel
from RayS.RayS import RayS

from .base import Attack


class RaySAttack(Attack):
    def __init__(self, model, args, **kwargs):
        super().__init__(model, args, **kwargs)
        num_queries = args["rays_num_queries"]
        self.model = GeneralTorchModel(
            model, n_class=self.num_classes, im_mean=None, im_std=None
        )
        order = np.inf if args["ord"] == "inf" else 2
        self.attack = RayS(self.model, epsilon=self.epsilon, order=order)
        self.num_queries = num_queries

    def run(self, imgs, labels):
        # x_adv, queries, adbd, succ = attack(data, label, query_limit)
        x_adv = self.attack(imgs, labels, query_limit=self.num_queries)[0]
        return x_adv
