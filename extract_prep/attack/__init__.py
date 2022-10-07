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

# from .rays import RaySAttack
from .bandit import BanditAttack
from .bayes import BayesOptAttack
from .boundary import BoundaryAttack
from .geoda import GeoDAAttack
from .hopskipjump import HopSkipJumpAttack
from .opt import OptAttack
from .qeba import QEBA
from .sign_opt import SignOptAttack
from .simba import SimBAAttack
from .square import SquareAttack
from .zoo import ZooAttack
from .fmn import FMNAttack

attack_dict = {
    # 'rays': RaySAttack,
    'hsj': HopSkipJumpAttack,
    'simba': SimBAAttack,
    'geoda': GeoDAAttack,
    'boundary': BoundaryAttack,
    'square': SquareAttack,
    'zoo': ZooAttack,
    'bandit': BanditAttack,
    'signopt': SignOptAttack,
    'bayes': BayesOptAttack,
    'opt': OptAttack,
    'qeba': QEBA,
    'fmn': FMNAttack,
}
