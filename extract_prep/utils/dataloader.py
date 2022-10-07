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

import torch
import torchvision.datasets as datasets
from torchvision import transforms


def get_dataloader(args):
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            args["data_dir"],
            transforms.Compose(
                [
                    transforms.Resize(int(args["orig_size"] * 256 / 224)),
                    transforms.CenterCrop(args["orig_size"]),
                    transforms.ToTensor(),
                ]
            ),
        ),
        batch_size=args["batch_size"],
        shuffle=True,
        num_workers=args["workers"],
        pin_memory=True,
    )
    return val_loader
