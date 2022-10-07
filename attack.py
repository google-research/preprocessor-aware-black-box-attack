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

import eagerpy
import foolbox.attacks
import numpy as np
import torch
import torchvision.models as models
from foolbox import PyTorchModel, accuracy, samples
from PIL import Image


def setup():
    global model, images, labels, fmodel, mmodel
    model = models.resnet18(pretrained=True).eval()
    model_fast = models.resnet18(pretrained=True).eval().cuda()

    class MyModel(torch.nn.Module):
        bounds = (0, 1)

        def __call__(self, x):
            x = (x[:, :, :224, :224] + x[:, :, 224:448, :224]) / 2
            # x = x[:, :, :224, :224]# + x[:, :, 224:448, :224])/2
            if "raw" in dir(x):
                x = x.raw
            with torch.no_grad():
                r = model_fast(x.cuda())
            print(r.argmax(1).cpu().numpy())
            return eagerpy.astensor(r.cpu())

    mmodel = MyModel()

    preprocessing = dict(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3
    )
    fmodel = PyTorchModel(mmodel, bounds=(0, 1), preprocessing=preprocessing)

    images, labels = samples(fmodel, dataset="imagenet", batchsize=32)


def run():
    print(labels.shape, images.shape)

    img = images.cpu()
    img = torch.cat([img, img], 2)

    fake_labels = mmodel(img.cuda()).raw.argmax(1).cpu()
    criteria = foolbox.criteria.Misclassification(fake_labels)
    attack = foolbox.attacks.HopSkipJump(steps=10)
    # attack = foolbox.attacks.BoundaryAttack(steps=3000)
    # img = torch.cat([img, img, img], 2)
    # img = torch.cat([img, img], 3)
    print(img.shape)

    out = attack.run(mmodel, img, criterion=criteria)

    print("Done")

    print("Original")
    print(mmodel(img.cuda()).raw.argmax(1).cpu().numpy())
    print("Adversarial")
    print(mmodel(out.cuda()).raw.argmax(1).cpu().numpy())
    print("Distortion")
    print(torch.sum((img - out) ** 2, (1, 2, 3)) ** 0.5)
    print((torch.sum((img - out) ** 2, (1, 2, 3)) ** 0.5).mean())


# 18.9 with actual image
# 30.1 with left half

if __name__ == "__main__":
    setup()
    run()
    exit(0)
