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

from typing import Tuple

import numpy as np
import torch


def normalize(
    x: torch.Tensor,
    batch: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize x in-place

    Args:
        x (torch.Tensor): Tensor to normalize
        batch (bool): First dimension of x is batch
    """
    if batch:
        norm = x.reshape(x.size(0), -1).norm(2, 1)
    else:
        norm = x.norm()
    for _ in range(x.ndim - 1):
        norm.unsqueeze_(-1)
    x /= (norm + 1e-9)
    if batch:
        norm = norm.view(x.size(0), 1)
    else:
        norm.squeeze_()
    return x, norm


class PytorchModel(object):
    def __init__(self, model, bounds):
        self.model = model
        self.model.eval()
        self.bounds = bounds
        self.num_queries = 0

    def predict(self, image):
        image = torch.clamp(image, self.bounds[0], self.bounds[1]).cuda()
        if len(image.size()) != 4:
            image = image.unsqueeze(0)
        output = self.model(image)
        self.num_queries += 1
        return output

    def predict_prob(self, image):
        with torch.no_grad():
            image = torch.clamp(image, self.bounds[0], self.bounds[1]).cuda()
            if len(image.size()) != 4:
                image = image.unsqueeze(0)
            output = self.model(image)
            self.num_queries += image.size(0)
        return output

    def predict_label(self, image, batch=True):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).type(torch.FloatTensor)
        image = torch.clamp(image, self.bounds[0], self.bounds[1]).cuda()
        if len(image.size()) != 4:
            image = image.unsqueeze(0)
            batch = False
        # image = Variable(image, volatile=True) # ?? not supported by latest pytorch
        with torch.no_grad():
            output = self.model(image)
            self.num_queries += image.size(0)
        # image = Variable(image, volatile=True) # ?? not supported by latest pytorch
        _, predict = torch.max(output.data, 1)
        if batch:
            return predict
        else:
            return predict[0]

    def predict_ensemble(self, image):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).type(torch.FloatTensor)
        image = torch.clamp(image, self.bounds[0], self.bounds[1]).cuda()
        if len(image.size()) != 4:
            image = image.unsqueeze(0)
        with torch.no_grad():
            output = self.model(image)
            output.zero_()
            for i in range(10):
                output += self.model(image)
                self.num_queries += image.size(0)

        _, predict = torch.max(output.data, 1)

        return predict[0]

    def get_num_queries(self):
        return self.num_queries

    # def get_infor_hard(self, dis_history, query_history):
    #    return torch.mean(dis_history,2,keepdim=True), torch.mean(query_history,2,keepdim=True)

    def get_gradient(self, loss):
        loss.backward()


def fine_grained_binary_search_local(model, x0, y0, theta, initial_lbd=1.0, tol=1e-5):
    nquery = 0
    lbd = initial_lbd

    if model.predict_label(x0 + lbd * theta) == y0:
        lbd_lo = lbd
        lbd_hi = lbd * 1.01
        nquery += 1
        while model.predict_label(x0 + lbd_hi * theta) == y0:
            lbd_hi *= 1.01
            nquery += 1
            if lbd_hi > 20:
                return float('inf'), nquery
    else:
        lbd_hi = lbd
        lbd_lo = lbd * 0.99
        nquery += 1
        while model.predict_label(x0 + lbd_lo * theta) != y0:
            lbd_lo *= 0.99
            nquery += 1

    # EDIT: fix bug that makes Sign-OPT stuck in while loop
    # while lbd_hi - lbd_lo > tol:
    diff = lbd_hi - lbd_lo
    while diff > tol:
        lbd_mid = (lbd_lo + lbd_hi) / 2
        # EDIT: add a break condition
        if lbd_mid == lbd_hi or lbd_mid == lbd_lo:
            break
        nquery += 1
        if model.predict_label(x0 + lbd_mid * theta) != y0:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
        # EDIT: This is to avoid numerical issue with gpu tensor when diff is small
        if diff <= lbd_hi - lbd_lo:
            break
        diff = lbd_hi - lbd_lo
    return lbd_hi, nquery


def fine_grained_binary_search(model, x0, y0, theta, initial_lbd, current_best):
    nquery = 0
    if initial_lbd > current_best:
        if model.predict_label(x0 + current_best * theta) == y0:
            nquery += 1
            return float('inf'), nquery
        lbd = current_best
    else:
        lbd = initial_lbd

    lbd_hi = lbd
    lbd_lo = 0.0

    # EDIT: This tol check has a numerical issue and may never quit (1e-5)
    while lbd_hi - lbd_lo > 1e-5:
        lbd_mid = (lbd_lo + lbd_hi) / 2
        # EDIT: add a break condition
        if lbd_mid == lbd_hi or lbd_mid == lbd_lo:
            break
        nquery += 1
        if model.predict_label(x0 + lbd_mid * theta) != y0:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid

    return lbd_hi, nquery


def fine_grained_binary_search_local_targeted(model, x0, t, theta,
                                              initial_lbd=1.0, tol=1e-5):
    # TODO: bug is not fixed here
    nquery = 0
    lbd = initial_lbd

    if model.predict_label(x0 + lbd * theta) != t:
        lbd_lo = lbd
        lbd_hi = lbd * 1.01
        nquery += 1
        while model.predict_label(x0 + lbd_hi * theta) != t:
            lbd_hi = lbd_hi * 1.01
            nquery += 1
            if lbd_hi > 100:
                return float('inf'), nquery
    else:
        lbd_hi = lbd
        lbd_lo = lbd * 0.99
        nquery += 1
        while model.predict_label(x0 + lbd_lo * theta) == t:
            lbd_lo = lbd_lo * 0.99
            nquery += 1

    while (lbd_hi - lbd_lo) > tol:
        lbd_mid = (lbd_lo + lbd_hi) / 2.0
        # EDIT: add a break condition
        if lbd_mid == lbd_hi or lbd_mid == lbd_lo:
            break
        nquery += 1
        if model.predict_label(x0 + lbd_mid * theta) == t:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid

    return lbd_hi, nquery


def fine_grained_binary_search_targeted(model, x0, t, theta,
                                        initial_lbd, current_best):
    nquery = 0
    if initial_lbd > current_best:
        if model.predict_label(x0 + current_best * theta) != t:
            nquery += 1
            return float('inf'), nquery
        lbd = current_best
    else:
        lbd = initial_lbd

    lbd_hi = lbd
    lbd_lo = 0.0

    while (lbd_hi - lbd_lo) > 1e-5:
        lbd_mid = (lbd_lo + lbd_hi) / 2.0
        # EDIT: add a break condition
        if lbd_mid == lbd_hi or lbd_mid == lbd_lo:
            break
        nquery += 1
        if model.predict_label(x0 + lbd_mid * theta) != t:
            lbd_lo = lbd_mid
        else:
            lbd_hi = lbd_mid
    return lbd_hi, nquery
