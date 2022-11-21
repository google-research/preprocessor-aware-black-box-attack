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

from autoattack.square import SquareAttack as SA

from .base import Attack


class SquareAttack(Attack):
    def __init__(self, model, args, subtract_steps=0, **kwargs):
        super().__init__(model, args, **kwargs)
        self.model = model
        if args["square_p"] is not None:
            p_init = args["square_p"]
        else:
            # Default: p.23 from paper (ImageNet)
            p_init = 0.1 if args["ord"] == "2" else 0.05
        self.attack = SA(
            self.model,
            p_init=p_init,
            n_queries=args["square_max_iter"] - subtract_steps,
            eps=self.epsilon,
            norm=f'L{args["ord"]}',
            n_restarts=1,
            seed=0,
            verbose=False,
            device="cuda",
            resc_schedule=True,  # rescale schedule according to n_queries
        )

    def run(self, imgs, labels, tgt=None):
        x_adv = self.attack.perturb(imgs.contiguous(), labels.contiguous())
        return x_adv
