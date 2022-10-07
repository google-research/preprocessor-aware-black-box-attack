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

import foolbox.attacks
from foolbox import PyTorchModel

from .base import Attack


class BoundaryAttack(Attack):
    def __init__(self, model, args, substract_steps=0, **kwargs):
        super().__init__(model, args, **kwargs)
        self.model = PyTorchModel(model, bounds=(0, 1), preprocessing=None)
        self.attack = foolbox.attacks.BoundaryAttack(
            init_attack=None,
            steps=args["max_iter"] - substract_steps,
            spherical_step=args["boundary_orth_step"],  # Default: 0.01
            source_step=args["boundary_step"],  # Default: 0.01
            source_step_convergance=1e-07,
            step_adaptation=1.5,
            update_stats_every_k=10,
        )

    def run(self, imgs, labels, tgt=None, **kwargs):
        if tgt is None:
            criteria = foolbox.criteria.Misclassification(labels)
            starting_points = None
        else:
            criteria = foolbox.criteria.TargetedMisclassification(tgt[1])
            starting_points = tgt[0]
        x_adv = self.attack.run(
            self.model,
            imgs,
            criterion=criteria,
            starting_points=starting_points,
        )
        return x_adv
