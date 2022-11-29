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

"""Base Attack class."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Attack(ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        config: dict[str, str | int | float],
        epsilon: float | None = None,
        input_size: int = 224,
        **kwargs,
    ) -> None:
        _ = kwargs  # Unused
        self.model: torch.nn.Module = model
        self.num_classes: int = config["num_classes"]
        self.epsilon: float = (
            epsilon if epsilon is not None else config["epsilon"]
        )
        self.input_size: int = input_size

    @abstractmethod
    def run(
        self,
        imgs: torch.Tensor,
        labels: torch.Tensor,
        tgt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return imgs
