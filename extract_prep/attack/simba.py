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
Code adapted from https://github.com/cg563/simple-blackbox-attack
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fftpack import idct

from .base import Attack


class SimBAAttack(Attack):
    def __init__(
        self, model, args, input_size=224, substract_steps=0, **kwargs
    ):
        super().__init__(model, args, **kwargs)
        self.model = nn.Sequential(model, nn.Softmax(1))
        self.image_size = input_size
        self.max_iters = args["simba_max_iter"] - substract_steps
        self.freq_dims = 28 if args["ord"] == "2" else 224
        self.stride = 7
        self.epsilon = 0.2
        self.linf_bound = 0 if args["ord"] == "2" else 0.05
        self.order = "strided" if args["ord"] == "2" else "rand"
        self.targeted = False
        self.pixel_attack = False  # if True, change freq_dims to 224
        self.log_every = 1000

    def run(self, imgs, labels):
        return self.simba_batch(imgs, labels)[0]

    def expand_vector(self, x, size):
        batch_size = x.size(0)
        x = x.view(-1, 3, size, size)
        z = torch.zeros(
            batch_size, 3, self.image_size, self.image_size, device=x.device
        )
        z[:, :, :size, :size] = x
        return z

    def get_probs(self, x, y):
        output = self.model(x.cuda())
        probs = torch.index_select(F.softmax(output, dim=-1), 1, y)
        return torch.diag(probs)

    def get_preds(self, x):
        output = self.model(x.cuda())
        _, preds = output.data.max(1)
        return preds

    # 20-line implementation of SimBA for single image input
    def simba_single(self, x, y, num_iters=10000, epsilon=0.2, targeted=False):
        """Not used (for reference)"""
        n_dims = x.view(1, -1).size(1)
        perm = torch.randperm(n_dims)
        x = x.unsqueeze(0)
        last_prob = self.get_probs(x, y)
        for i in range(num_iters):
            diff = torch.zeros(n_dims)
            diff[perm[i]] = epsilon
            left_prob = self.get_probs((x - diff.view(x.size())).clamp(0, 1), y)
            if targeted != (left_prob < last_prob):
                x = (x - diff.view(x.size())).clamp(0, 1)
                last_prob = left_prob
            else:
                right_prob = self.get_probs(
                    (x + diff.view(x.size())).clamp(0, 1), y
                )
                if targeted != (right_prob < last_prob):
                    x = (x + diff.view(x.size())).clamp(0, 1)
                    last_prob = right_prob
            if i % 10 == 0:
                print(last_prob)
        return x.squeeze()

    # runs simba on a batch of images <images_batch> with true labels (for untargeted attack) or target labels
    # (for targeted attack) <labels_batch>
    def simba_batch(self, images_batch, labels_batch):
        device = images_batch.device
        batch_size = images_batch.size(0)
        image_size = images_batch.size(2)
        assert self.image_size == image_size
        # sample a random ordering for coordinates independently per batch element
        if self.order == "rand":
            indices = torch.randperm(3 * self.freq_dims * self.freq_dims)[
                : self.max_iters
            ]
        elif self.order == "diag":
            indices = diagonal_order(image_size, 3)[: self.max_iters]
        elif self.order == "strided":
            indices = block_order(
                image_size, 3, initial_size=self.freq_dims, stride=self.stride
            )[: self.max_iters]
        else:
            indices = block_order(image_size, 3)[: self.max_iters]
        if self.order == "rand":
            expand_dims = self.freq_dims
        else:
            expand_dims = image_size
        n_dims = 3 * expand_dims * expand_dims
        x = torch.zeros((batch_size, n_dims), device=device)
        # logging tensors
        probs = torch.zeros((batch_size, self.max_iters), device=device)
        succs = torch.zeros_like(probs)
        queries = torch.zeros_like(probs)
        l2_norms = torch.zeros_like(probs)
        linf_norms = torch.zeros_like(probs)
        prev_probs = self.get_probs(images_batch, labels_batch)
        preds = self.get_preds(images_batch)
        if self.pixel_attack:

            def trans(z):
                return z

        else:

            def trans(z):
                return block_idct(
                    z, block_size=image_size, linf_bound=self.linf_bound
                )

        remaining_indices = torch.arange(0, batch_size, device=device).long()
        for k in range(self.max_iters):
            dim = indices[k]
            expanded = (
                images_batch[remaining_indices]
                + trans(self.expand_vector(x[remaining_indices], expand_dims))
            ).clamp(0, 1)
            perturbation = trans(self.expand_vector(x, expand_dims))
            l2_norms[:, k] = perturbation.view(batch_size, -1).norm(2, 1)
            linf_norms[:, k] = perturbation.view(batch_size, -1).abs().max(1)[0]
            preds_next = self.get_preds(expanded)
            preds[remaining_indices] = preds_next
            if self.targeted:
                remaining = preds.ne(labels_batch)
            else:
                remaining = preds.eq(labels_batch)
            # check if all images are misclassified and stop early
            if remaining.sum() == 0:
                adv = (
                    images_batch + trans(self.expand_vector(x, expand_dims))
                ).clamp(0, 1)
                probs_k = self.get_probs(adv, labels_batch)
                probs[:, k:] = probs_k.unsqueeze(1).repeat(
                    1, self.max_iters - k
                )
                succs[:, k:] = torch.ones(
                    batch_size, self.max_iters - k, device=device
                )
                queries[:, k:] = torch.zeros(
                    batch_size, self.max_iters - k, device=device
                )
                break
            remaining_indices = torch.arange(0, batch_size, device=device)[
                remaining
            ].long()
            if k > 0:
                succs[:, k - 1] = ~remaining
            diff = torch.zeros(remaining.sum(), n_dims, device=device)
            diff[:, dim] = self.epsilon
            left_vec = x[remaining_indices] - diff
            right_vec = x[remaining_indices] + diff
            # trying negative direction
            adv = (
                images_batch[remaining_indices]
                + trans(self.expand_vector(left_vec, expand_dims))
            ).clamp(0, 1)
            left_probs = self.get_probs(adv, labels_batch[remaining_indices])
            queries_k = torch.zeros(batch_size, device=device)
            # increase query count for all images
            queries_k[remaining_indices] += 1
            if self.targeted:
                improved = left_probs.gt(prev_probs[remaining_indices])
            else:
                improved = left_probs.lt(prev_probs[remaining_indices])
            # only increase query count further by 1 for images that did not improve in adversarial loss
            if improved.sum() < remaining_indices.size(0):
                queries_k[remaining_indices[~improved]] += 1
            # try positive directions
            adv = (
                images_batch[remaining_indices]
                + trans(self.expand_vector(right_vec, expand_dims))
            ).clamp(0, 1)
            right_probs = self.get_probs(adv, labels_batch[remaining_indices])
            if self.targeted:
                right_improved = right_probs.gt(
                    torch.max(prev_probs[remaining_indices], left_probs)
                )
            else:
                right_improved = right_probs.lt(
                    torch.min(prev_probs[remaining_indices], left_probs)
                )
            probs_k = prev_probs.clone()
            # update x depending on which direction improved
            if improved.sum() > 0:
                left_indices = remaining_indices[improved]
                left_mask_remaining = improved.unsqueeze(1).repeat(1, n_dims)
                x[left_indices] = left_vec[left_mask_remaining].view(-1, n_dims)
                probs_k[left_indices] = left_probs[improved]
            if right_improved.sum() > 0:
                right_indices = remaining_indices[right_improved]
                right_mask_remaining = right_improved.unsqueeze(1).repeat(
                    1, n_dims
                )
                x[right_indices] = right_vec[right_mask_remaining].view(
                    -1, n_dims
                )
                probs_k[right_indices] = right_probs[right_improved]
            probs[:, k] = probs_k
            queries[:, k] = queries_k
            prev_probs = probs[:, k]
            if (k + 1) % self.log_every == 0 or k == self.max_iters - 1:
                print(
                    "Iteration %d: queries = %.4f, prob = %.4f, remaining = %.4f"
                    % (
                        k + 1,
                        queries.sum(1).mean(),
                        probs[:, k].mean(),
                        remaining.float().mean(),
                    )
                )
        expanded = (
            images_batch + trans(self.expand_vector(x, expand_dims))
        ).clamp(0, 1)
        preds = self.get_preds(expanded)
        if self.targeted:
            remaining = preds.ne(labels_batch)
        else:
            remaining = preds.eq(labels_batch)
        succs[:, self.max_iters - 1] = ~remaining
        return expanded, probs, succs, queries, l2_norms, linf_norms


def diagonal_order(image_size, channels):
    # defines a diagonal order
    # order is fixed across diagonals but are randomized across channels and within the diagonal
    # e.g.
    # [1, 2, 5]
    # [3, 4, 8]
    # [6, 7, 9]
    x = torch.arange(0, image_size).cumsum(0)
    order = torch.zeros(image_size, image_size)
    for i in range(image_size):
        order[i, : (image_size - i)] = i + x[i:]
    for i in range(1, image_size):
        reverse = order[image_size - i - 1].index_select(
            0, torch.LongTensor([i for i in range(i - 1, -1, -1)])
        )
        order[i, (image_size - i) :] = image_size * image_size - 1 - reverse
    if channels > 1:
        order_2d = order
        order = torch.zeros(channels, image_size, image_size)
        for i in range(channels):
            order[i, :, :] = 3 * order_2d + i
    return order.view(1, -1).squeeze().long().sort()[1]


def block_order(image_size, channels, initial_size=1, stride=1):
    # defines a block order, starting with top-left (initial_size x initial_size) submatrix
    # expanding by stride rows and columns whenever exhausted
    # randomized within the block and across channels
    # e.g. (initial_size=2, stride=1)
    # [1, 3, 6]
    # [2, 4, 9]
    # [5, 7, 8]
    order = torch.zeros(channels, image_size, image_size)
    total_elems = channels * initial_size * initial_size
    perm = torch.randperm(total_elems)
    order[:, :initial_size, :initial_size] = perm.view(
        channels, initial_size, initial_size
    )
    for i in range(initial_size, image_size, stride):
        num_elems = channels * (2 * stride * i + stride * stride)
        perm = torch.randperm(num_elems) + total_elems
        num_first = channels * stride * (stride + i)
        order[:, : (i + stride), i : (i + stride)] = perm[:num_first].view(
            channels, -1, stride
        )
        order[:, i : (i + stride), :i] = perm[num_first:].view(
            channels, stride, -1
        )
        total_elems += num_elems
    return order.view(1, -1).squeeze().long().sort()[1]


def block_idct(x, block_size=8, masked=False, ratio=0.5, linf_bound=0.0):
    # applies IDCT to each block of size block_size
    z = torch.zeros_like(x)
    num_blocks = int(x.size(2) / block_size)
    mask = np.zeros((x.size(0), x.size(1), block_size, block_size))
    if type(ratio) != float:
        for i in range(x.size(0)):
            mask[
                i, :, : int(block_size * ratio[i]), : int(block_size * ratio[i])
            ] = 1
    else:
        mask[:, :, : int(block_size * ratio), : int(block_size * ratio)] = 1
    for i in range(num_blocks):
        for j in range(num_blocks):
            submat = (
                x[
                    :,
                    :,
                    (i * block_size) : ((i + 1) * block_size),
                    (j * block_size) : ((j + 1) * block_size),
                ]
                .cpu()
                .numpy()
            )
            if masked:
                submat = submat * mask
            z[
                :,
                :,
                (i * block_size) : ((i + 1) * block_size),
                (j * block_size) : ((j + 1) * block_size),
            ] = torch.from_numpy(
                idct(idct(submat, axis=3, norm="ortho"), axis=2, norm="ortho")
            ).to(
                x.device
            )
    if linf_bound > 0:
        return z.clamp(-linf_bound, linf_bound)
    else:
        return z
