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

"""Neural super-resolution preprocessor."""

from __future__ import annotations

from pathlib import Path

import torch
from mmedit.apis import init_model
from torch import nn
from torchvision import transforms

from attack_prep.preprocessor.base import Preprocessor
from attack_prep.preprocessor.util import BICUBIC

_CONFIG = {
    "edsr": "edsr_x4c64b16_1xb16-300k_div2k.py",
    # "esrgan": "esrgan_x4c64b23g32_1xb16-400k_div2k.py",
    "esrgan": "esrgan_psnr-x4c64b23g32_1xb16-1000k_div2k.py",
    "real_esrgan": "realesrnet_c64b23g32_4xb12-lr2e-4-1000k_df2k-ost.py",
    # "ttsr": "ttsr-gan_x4c64b16_1xb9-500k_CUFED.py",
}

# pylint: disable=line-too-long
_CHECKPOINT = {
    "edsr": "https://download.openmmlab.com/mmediting/restorers/edsr/edsr_x4c64b16_1x16_300k_div2k_20200608-3c2af8a3.pth",
    # "esrgan": "https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_x4c64b23g32_1x16_400k_div2k_20200508-f8ccaf3b.pth",
    "esrgan": "https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_psnr_x4c64b23g32_1x16_1000k_div2k_20200420-bf5c993c.pth",
    "real_esrgan": "https://download.openmmlab.com/mmediting/restorers/real_esrgan/realesrnet_c64b23g32_12x4_lr2e-4_1000k_df2k_ost_20210816-4ae3b5a4.pth",
    # "ttsr": "https://download.openmmlab.com/mmediting/restorers/ttsr/ttsr-gan_x4_c64b16_g1_500k_CUFED_20210626-2ab28ca0.pth",
}


class Restore(nn.Module):
    """Restore image."""

    def __init__(self, model: nn.Module) -> None:
        """Initialize Restore module.

        Args:
            model: Model used to restore image.
        """
        super().__init__()
        self._model: nn.Module = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Restore inputs using MMEditing API."""
        outputs: torch.Tensor = self._model(mode="tensor", inputs=inputs)
        # This clamp is needed. Sometimes model outputs images out of range.
        outputs.clamp_(0, 1)
        return outputs


class SuperResolution(Preprocessor):
    """Super-resolution preprocessor."""

    def __init__(
        self,
        params: dict[str, str | int | float],
        **kwargs,
    ) -> None:
        """Initialize SuperResolution.

        Args:
            params: super-resolution preprocessor params.
            input_size: Image input size.
        """
        super().__init__(params, **kwargs)
        # For now, all models are 4x SR
        self.output_size: int = self.input_size * 4
        model_name: str = params["sr_model"]
        config_path: Path = Path(params["sr_config_path"]) / model_name
        config_path /= _CONFIG[model_name]
        config_path = config_path.expanduser()

        model = init_model(str(config_path), checkpoint=_CHECKPOINT[model_name])
        model.eval().to("cuda")
        for param in model.parameters():
            param.requires_grad = False

        self.prep: nn.Module = Restore(model)
        self.inv_prep: nn.Module = transforms.Resize(
            self.input_size, interpolation=BICUBIC
        )
