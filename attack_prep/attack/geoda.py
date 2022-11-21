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
import torch
import torch.nn.functional as F
from art.attacks.evasion import GeoDA

from .base import Attack
from .util import setup_art


class GeoDAAttack(Attack):
    def __init__(
        self, model, args, input_size=224, substract_steps=0, **kwargs
    ):
        super().__init__(model, args, **kwargs)
        self.model = setup_art(args, model, input_size)
        order = np.inf if args["ord"] == "inf" else 2
        # Parameters taken from https://github.com/thisisalirah/GeoDA/blob/master/GeoDA.py
        self.attack = GeoDA(
            self.model,
            batch_size=args["batch_size"],
            norm=order,
            sub_dim=75,
            max_iter=args["geoda_max_iter"] - substract_steps,
            bin_search_tol=0.0001,
            lambda_param=0.6,
            sigma=0.0002,
            verbose=args["debug"] or args["verbose"],
        )

    def run(self, imgs, labels):
        y = F.one_hot(labels, num_classes=self.num_classes).cpu().numpy()
        x_adv = self.attack.generate(imgs.cpu().numpy(), y=y)
        return torch.from_numpy(x_adv).cuda()
