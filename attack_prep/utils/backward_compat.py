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

"""Script to ensure backward compatibility with Nvidia PyTorch docker."""

import sys

from packaging import version

# Calling subprocess.check_output() with python version 3.8.10 or lower will
# raise NotADirectoryError. When torch calls this to call hipconfig, it does
# not catch this exception but only FileNotFoundError or PermissionError.
# This hack makes sure that correct exception is raised.
if version.parse(sys.version.split()[0]) <= version.parse("3.8.10"):
    import subprocess

    def _hacky_subprocess_fix(*args, **kwargs):
        raise FileNotFoundError(
            "Hacky exception. If this interferes with your workflow, consider "
            "using python >= 3.8.10 or simply try to comment this out."
        )

    subprocess.check_output = _hacky_subprocess_fix

    # This floating point (core dumped) error is caused by this line in mmengine
    # https://github.com/open-mmlab/mmengine/blob/main/mmengine/model/weight_init.py#L645
    # Specifically, the function `erfinv_(x)`. If you don't face this problem,
    # you can safely comment this out.
    import torch
    from scipy import special

    def _hacky_erfinv(self):
        erfinv = torch.from_numpy(special.erfinv(self.numpy()))
        self.zero_()
        self.add_(erfinv)

    torch.Tensor.erfinv_ = _hacky_erfinv
