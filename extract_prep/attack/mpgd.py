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

"""
Code is adapted from 
https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/mifgsm.py
"""
import torch
import torch.nn as nn

from .base import Attack


class MomentumPGD(Attack):
    r"""
    MI-FGSM in the paper 'Boosting Adversarial Attacks with Momentum'
    [https://arxiv.org/abs/1710.06081]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        decay (float): momentum factor. (Default: 1.0)
        steps (int): number of iterations. (Default: 5)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.MIFGSM(model, eps=8/255, steps=5, decay=1.0)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, args, **kwargs):
        super().__init__(model, args, **kwargs)
        self.eps = args["epsilon"]
        self.steps = 300
        self.decay = 1.0
        self.step_size = 0.001
        self._targeted = False
        self.loss = nn.CrossEntropyLoss()

    def run(self, images, labels):
        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        momentum = torch.zeros_like(images, requires_grad=False)
        adv_images = images.clone()

        for _ in range(self.steps):
            adv_images.requires_grad_()
            outputs = self.model(adv_images)

            # Calculate loss
            if self._targeted:
                cost = -self.loss(outputs, target_labels)
            else:
                cost = self.loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images)[0].detach()
            grad /= torch.abs(grad).mean(dim=(1, 2, 3), keepdim=True) + 1e-6
            grad += momentum * self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.step_size * grad.sign()
            delta = torch.clamp(
                adv_images - images, min=-self.eps, max=self.eps
            )
            adv_images = torch.clamp(images + delta, min=0, max=1)

        return adv_images
