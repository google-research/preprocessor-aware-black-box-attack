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

"""Wrapper around Foolbox HopSkipJump attack implementation."""

import foolbox
import numpy as np
from foolbox import PyTorchModel
from typing import Any, Optional

from extract_prep.attack.base import Attack
from extract_prep.attack.hopskipjump.hop_skip_jump import HopSkipJump


class HopSkipJumpAttack(Attack):
    def __init__(
        self,
        model,
        args,
        substract_steps=0,
        preprocess=None,
        smart_noise: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(model, args, **kwargs)
        self.model = PyTorchModel(model, bounds=(0, 1), preprocessing=None)
        # (1) gradient_eval_steps is min([initial_gradient_eval_steps *
        # math.sqrt(step + 1), max_gradient_eval_steps]) (L149)
        # (2) step size search also uses a few more queries. Geometric search
        # has while loop and can't be pre-determined (L166)
        # (3) binary search (L184) also has while loop

        # Approximate (upper bound of) `steps`
        # \sum_{i=1}^{steps} (sqrt(i) * init_grad_steps) <= max_iter
        max_iter = args["max_iter"] - substract_steps - 51  # 51 for init attack
        iters = np.sqrt(np.arange(100)) * args["hsj_init_grad_steps"]
        iters = np.cumsum(np.minimum(iters, args["hsj_max_grad_steps"]))
        steps = np.where(iters <= max_iter)[0][-1]

        self.attack = HopSkipJump(
            max_queries=args["max_iter"],
            steps=steps,
            initial_gradient_eval_steps=args["hsj_init_grad_steps"],
            max_gradient_eval_steps=args["hsj_max_grad_steps"],
            gamma=args["hsj_gamma"],
            constraint=f'l{args["ord"]}',
            verbose=args["verbose"],
            preprocess=preprocess,
            smart_noise=smart_noise,
            norm_rv=args["hsj_norm_rv"],
            prep_backprop=args["prep_backprop"],
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
