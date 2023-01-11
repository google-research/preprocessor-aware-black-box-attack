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

from attack_prep.preprocessor.base import Identity, Preprocessor

_ALLOWED_MODELS = {"swinir", "restormer", "nafnet"}

_CONFIG = {
    "swinir_dn15": "swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN15.py",
    "swinir_dn25": "swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN25.py",
    "swinir_dn50": "swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN50.py",
    "swinir_car10": "swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR10.py",
    "swinir_car20": "swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR20.py",
    "swinir_car30": "swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR30.py",
    "swinir_car40": "swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR40.py",
    # Restormer (CVPR22): https://arxiv.org/abs/2111.09881
    "restormer_rain": "restormer_official_rain13k.py",
    "restormer_motion": "restormer_official_gopro.py",
    "restormer_blur-single": "restormer_official_dpdd-single.py",
    "restormer_blur-dual": "restormer_official_dpdd-dual.py",
    "restormer_gn15": "restormer_official_dfwb-color-sigma15.py",
    "restormer_gn25": "restormer_official_dfwb-color-sigma25.py",
    "restormer_gn50": "restormer_official_dfwb-color-sigma50.py",
    "restormer_real": "restormer_official_sidd.py",
    # NAFNet (ECCV22): https://arxiv.org/abs/2204.04676
    "nafnet_real": "nafnet_c64eb2248mb12db2222_8xb8-lr1e-3-400k_sidd.py",
    "nafnet_motion": "nafnet_c64eb11128mb1db1111_8xb8-lr1e-3-400k_gopro.py",
}

# pylint: disable=line-too-long
_CHECKPOINT = {
    "swinir_dn15": "https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN15-c74a2cee.pth",
    "swinir_dn25": "https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN25-df2b1c0c.pth",
    "swinir_dn50": "https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN50-e369874c.pth",
    "swinir_car10": "https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR10-09aafadc.pth",
    "swinir_car20": "https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR20-b8a42b5e.pth",
    "swinir_car30": "https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR30-e9fe6859.pth",
    "swinir_car40": "https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR40-5b77a6e6.pth",
    "restormer_rain": "https://download.openmmlab.com/mmediting/restormer/restormer_official_rain13k-2be7b550.pth",
    "restormer_motion": "https://download.openmmlab.com/mmediting/restormer/restormer_official_gopro-db7363a0.pth",
    "restormer_blur-single": "https://download.openmmlab.com/mmediting/restormer/restormer_official_dpdd-single-6bc31582.pth",
    "restormer_blur-dual": "https://download.openmmlab.com/mmediting/restormer/restormer_official_dpdd-dual-52c94c00.pth",
    "restormer_gn15": "https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma15-012ceb71.pth",
    "restormer_gn25": "https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma25-e307f222.pth",
    "restormer_gn50": "https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma50-a991983d.pth",
    "restormer_real": "https://download.openmmlab.com/mmediting/restormer/restormer_official_sidd-9e7025db.pth",
    "nafnet_real": "https://download.openmmlab.com/mmediting/nafnet/NAFNet-SIDD-midc64.pth",
    "nafnet_motion": "https://download.openmmlab.com/mmediting/nafnet/NAFNet-GoPro-midc64.pth",
}


class Denoise(nn.Module):
    """Denoise image."""

    def __init__(self, model: nn.Module) -> None:
        """Initialize Denoise module.

        Args:
            model: Model used to restore image.
        """
        super().__init__()
        self._model: nn.Module = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Restore inputs using MMEditing API."""
        outputs: torch.Tensor = self._model(
            mode="tensor", inputs=inputs.float()
        )
        # This clamp is needed. Sometimes model outputs images out of range.
        outputs.clamp_(0, 1)
        return outputs


class Denoiser(Preprocessor):
    """Image-denoising preprocessor."""

    def __init__(
        self,
        params: dict[str, str | int | float],
        **kwargs,
    ) -> None:
        """Initialize Denoiser.

        Args:
            params: super-resolution preprocessor params.
            input_size: Image input size.
        """
        super().__init__(params, **kwargs)
        self.output_size: int = self.input_size
        model_type: str = params["denoise_model"]
        if model_type not in _ALLOWED_MODELS:
            raise NotImplementedError(
                f"denoise_model {model_type} is not implemented! Only support "
                f"{_ALLOWED_MODELS}."
            )
        denoise_mode: str = params["denoise_mode"]
        model_name: str = f"{model_type}_{denoise_mode}"
        if model_name not in _CONFIG:
            raise NotImplementedError(
                f"Combination of denoise_model and denoise_mode {model_name} "
                f"is not implemented! Only support {_CONFIG.keys()}."
            )
        config_path: Path = Path(params["denoise_config_path"]) / model_type
        config_path /= _CONFIG[model_name]
        config_path = config_path.expanduser()

        # Warnining when loading real_esrgan is expected, can be subpressed by
        # this hack: https://github.com/open-mmlab/mmediting/issues/1439.
        model = init_model(str(config_path), checkpoint=_CHECKPOINT[model_name])
        model.eval().to("cuda")
        for param in model.parameters():
            param.requires_grad = False

        self.prep: nn.Module = Denoise(model)
        self.inv_prep: nn.Module = Identity()
