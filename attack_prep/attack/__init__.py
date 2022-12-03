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

from attack_prep.attack.bandit import BanditAttack
from attack_prep.attack.boundary import BoundaryAttack
from attack_prep.attack.fmn import FMNAttack
from attack_prep.attack.hopskipjump import HopSkipJumpAttack
from attack_prep.attack.opt import OptAttack
from attack_prep.attack.qeba import QEBA
from attack_prep.attack.sign_opt import SignOptAttack
from attack_prep.attack.simba import SimBAAttack
from attack_prep.attack.square import SquareAttack
from attack_prep.attack.zoo import ZooAttack

# from .rays import RaySAttack
# from .bayes import BayesOptAttack
# from .geoda import GeoDAAttack


ATTACK_DICT = {
    # 'rays': RaySAttack,
    "hsj": HopSkipJumpAttack,
    "simba": SimBAAttack,
    # 'geoda': GeoDAAttack,
    "boundary": BoundaryAttack,
    "square": SquareAttack,
    "zoo": ZooAttack,
    "bandit": BanditAttack,
    "signopt": SignOptAttack,
    # 'bayes': BayesOptAttack,
    "opt": OptAttack,
    "qeba": QEBA,
    "fmn": FMNAttack,
}
