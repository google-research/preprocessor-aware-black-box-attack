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
Code is adapted from the original repo
https://github.com/MadryLab/blackbox-bandits/blob/master/src/main.py
"""
import torch as ch
from attack_prep.preprocessor.quantize import Quant
from torch.nn.modules import Upsample

from .base import Attack


def norm(t):
    assert len(t.shape) == 4
    norm_vec = ch.sqrt(t.pow(2).sum(dim=[1, 2, 3])).view(-1, 1, 1, 1)
    norm_vec += (norm_vec == 0).float() * 1e-8
    return norm_vec


###
# Different optimization steps
# All take the form of func(x, g, lr)
# eg: exponentiated gradients
# l2/linf: projected gradient descent
###


def eg_step(x, g, lr):
    real_x = (x + 1) / 2  # from [-1, 1] to [0, 1]
    pos = real_x * ch.exp(lr * g)
    neg = (1 - real_x) * ch.exp(-lr * g)
    new_x = pos / (pos + neg)
    return new_x * 2 - 1


def linf_step(x, g, lr):
    return x + lr * ch.sign(g)


def l2_prior_step(x, g, lr):
    new_x = x + lr * g / norm(g)
    norm_new_x = norm(new_x)
    norm_mask = (norm_new_x < 1.0).float()
    return new_x * norm_mask + (1 - norm_mask) * new_x / norm_new_x


def gd_prior_step(x, g, lr):
    return x + lr * g


def l2_image_step(x, g, lr):
    return x + lr * g / norm(g)


##
# Projection steps for l2 and linf constraints:
# All take the form of func(new_x, old_x, epsilon)
##


def l2_proj(image, eps):
    orig = image.clone()

    def proj(new_x):
        delta = new_x - orig
        out_of_bounds_mask = (norm(delta) > eps).float()
        x = (orig + eps * delta / norm(delta)) * out_of_bounds_mask
        x += new_x * (1 - out_of_bounds_mask)
        return x

    return proj


def linf_proj(image, eps):
    orig = image.clone()

    def proj(new_x):
        return orig + ch.clamp(new_x - orig, -eps, eps)

    return proj


class Parameters:
    """
    Parameters class, just a nice way of accessing a dictionary
    > ps = Parameters({"a": 1, "b": 3})
    > ps.A # returns 1
    > ps.B # returns 3
    """

    def __init__(self, params):
        self.params = params

    def __getattr__(self, x):
        return self.params[x.lower()]


class BanditAttack(Attack):
    def __init__(self, model, args, substract_steps=0, **kwargs):
        super().__init__(model, args, **kwargs)
        self.model = model
        self.order = f"l{args['ord']}"
        # Default parameters https://github.com/MadryLab/blackbox-bandits/tree/master/src/configs
        params = {
            "2": {
                "fd_eta": 0.01,
                "image_lr": 0.5,
                "online_lr": 0.1,
                "exploration": 0.01,
            },  # eps 5
            "inf": {
                "fd_eta": 0.1,
                "image_lr": 0.01,
                "online_lr": 100,
                "exploration": 1.0,
            },  # eps 0.05
        }[args["ord"]]
        self.params = Parameters(
            {
                "max_queries": args["bandit_max_iter"],
                "gradient_iters": 1,
                "tile_size": 50,
                "nes": False,
                "tiling": True,
                "quantize": False,
                "log_progress": True,
                **params,
            }
        )
        # self.params = Parameters({
        #     "fd_eta": args['bandit_fd_eta'],
        #     "max_queries": args['bandit_max_iter'] - substract_steps,
        #     "image_lr": args['bandit_image_lr'],
        #     "online_lr": 0.1,
        #     "exploration": args['bandit_fd_eta'],
        #     "gradient_iters": 1,
        #     "tile_size": 50,
        #     "nes": False,
        #     "tiling": True,
        #     # "quantize": False,
        #     "quantize": args['preprocess'] == 'quantize',
        #     "log_progress": True,
        # })
        self.quantize = Quant(8)

    def make_adversarial_examples(self, image, true_label):
        """
        The main process for generating adversarial examples with priors.
        """
        # Initial setup
        args = self.params
        batch_size = image.size(0)
        device = image.device
        input_size = image.size(-1)
        prior_size = input_size if not args.tiling else args.tile_size
        upsampler = Upsample(size=(input_size, input_size))
        total_queries = ch.zeros(batch_size, device=device)
        prior = ch.zeros(batch_size, 3, prior_size, prior_size, device=device)
        dim = prior.nelement() / batch_size
        prior_step = gd_prior_step if self.order == "l2" else eg_step
        image_step = l2_image_step if self.order == "l2" else linf_step
        proj_maker = l2_proj if self.order == "l2" else linf_proj
        proj_step = proj_maker(image, self.epsilon)

        # Loss function
        criterion = ch.nn.CrossEntropyLoss(reduction="none")

        def L(x):
            return criterion(self.model(x), true_label)

        # Original classifications
        orig_images = image.clone()
        orig_classes = self.model(image).argmax(1).cuda()
        correct_classified_mask = (orig_classes == true_label).float()
        # total_ims = correct_classified_mask.sum()
        not_dones_mask = correct_classified_mask.clone()

        t = 0
        while not ch.any(total_queries > args.max_queries):
            t += args.gradient_iters * 2
            if t >= args.max_queries:
                break
            if not args.nes:
                # Updating the prior:
                # Create noise for exporation, estimate the gradient, and take a PGD step
                exp_noise = (
                    args.exploration * ch.randn_like(prior) / (dim**0.5)
                )
                # Query deltas for finite difference estimator
                q1 = upsampler(prior + exp_noise)
                q2 = upsampler(prior - exp_noise)
                # Loss points for finite difference estimator
                l1 = L(
                    image + args.fd_eta * q1 / norm(q1)
                )  # L(prior + c*noise)
                l2 = L(
                    image + args.fd_eta * q2 / norm(q2)
                )  # L(prior - c*noise)
                # Finite differences estimate of directional derivative
                est_deriv = (l1 - l2) / (args.fd_eta * args.exploration)
                # 2-query gradient estimate
                est_grad = est_deriv.view(-1, 1, 1, 1) * exp_noise
                # Update the prior with the estimated gradient
                prior = prior_step(prior, est_grad, args.online_lr)
            else:
                prior = ch.zeros_like(image)
                for _ in range(args.gradient_iters):
                    exp_noise = ch.randn_like(image) / (dim**0.5)
                    est_deriv = (
                        L(image + args.fd_eta * exp_noise)
                        - L(image - args.fd_eta * exp_noise)
                    ) / args.fd_eta
                    prior += est_deriv.view(-1, 1, 1, 1) * exp_noise

                # Preserve images that are already done,
                # Unless we are specifically measuring gradient estimation
                prior = prior * not_dones_mask.view(-1, 1, 1, 1)

            # Update the image:
            # take a pgd step using the prior
            new_im = image_step(
                image,
                upsampler(prior * correct_classified_mask.view(-1, 1, 1, 1)),
                args.image_lr,
            )
            image = proj_step(new_im)
            if args.quantize:
                image = self.quantize(image)
            image = ch.clamp(image, 0, 1)
            # if self.order == 'l2':
            #     if not ch.all(norm(image - orig_images) <= self.epsilon + 1e-3):
            #         pdb.set_trace()
            # else:
            #     if not (image - orig_images).max() <= self.epsilon + 1e-3:
            #         pdb.set_trace()

            # Continue query count
            total_queries += 2 * args.gradient_iters * not_dones_mask
            not_dones_mask = not_dones_mask * (
                (self.model(image).argmax(1) == true_label).float()
            )

            # Logging stuff
            # new_losses = L(image)
            success_mask = (1 - not_dones_mask) * correct_classified_mask
            num_success = success_mask.sum()
            current_success_rate = (
                (num_success / correct_classified_mask.sum()).cpu().item()
            )
            # success_queries = ((success_mask*total_queries).sum()/num_success).cpu().item()
            # not_done_loss = ((new_losses*not_dones_mask).sum()/not_dones_mask.sum()).cpu().item()
            # max_curr_queries = total_queries.max().cpu().item()
            # if args.log_progress:
            #     print("Queries: %d | Success rate: %f | Average queries: %f" %
            #           (max_curr_queries, current_success_rate, success_queries))

            if current_success_rate == 1.0:
                break

        # return {
        #     'average_queries': success_queries,
        #     'num_correctly_classified': correct_classified_mask.sum().cpu().item(),
        #     'success_rate': current_success_rate,
        #     'images_orig': orig_images.cpu().numpy(),
        #     'images_adv': image.cpu().numpy(),
        #     'all_queries': total_queries.cpu().numpy(),
        #     'correctly_classified': correct_classified_mask.cpu().numpy(),
        #     'success': success_mask.cpu().numpy()
        # }
        print("# queries: ", total_queries.cpu().numpy())

        return image

    def run(self, imgs, labels, tgt=None):
        x_adv = self.make_adversarial_examples(
            imgs.contiguous(), labels.contiguous()
        )
        return x_adv
