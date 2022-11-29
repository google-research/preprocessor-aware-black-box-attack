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
import torchvision.transforms.functional as TF
from torchjpeg.dct import block_idct, blockify, deblockify

from .pca_generator import PCAGenerator

# from .dct_generator import DCTGenerator
# from .resize_generator import ResizeGenerator


def load_pgen(pgen_type):
    if pgen_type == "naive":
        p_gen = None
    elif "resize" in pgen_type:
        p_gen = ResizeGenerator(factor=float(pgen_type[6:]))
    elif pgen_type == "DCT2352":
        p_gen = DCTGenerator(factor=8.0)
    elif pgen_type == "DCT4107":
        p_gen = DCTGenerator(factor=6.0)
    elif pgen_type == "DCT9408":
        p_gen = DCTGenerator(factor=4.0)
    elif pgen_type == "DCT16428":
        p_gen = DCTGenerator(factor=3.0)
    # elif pgen_type == 'NNGen':
    #     p_gen = NNGenerator(N_b=30, n_channels=3, gpu=args.use_gpu)
    #     p_gen.load_state_dict(torch.load('nn_gen_30_imagenet.model'))
    # elif pgen_type == 'ENC':
    #     p_gen = UNet(n_channels=3)
    #     p_gen.load_state_dict(torch.load('unet.model', map_location='cpu'))
    elif pgen_type == "PCA1000":
        p_gen = PCAGenerator(N_b=1000, approx=True)
        p_gen.load("pca_gen_1000_imagenet.npy")
    elif pgen_type == "PCA5000":
        p_gen = PCAGenerator(N_b=5000, approx=True)
        p_gen.load("pca_gen_5000_imagenet.npy")
    elif pgen_type == "PCA9408":
        p_gen = PCAGenerator(N_b=9408, approx=True)
        p_gen.load("pca_gen_9408_imagenet_avg.npy")
    elif pgen_type == "PCA2352basis":
        p_gen = PCAGenerator(N_b=2352, approx=True, basis_only=True)
        p_gen.load("pca_gen_2352_imagenet_avg.npy")
    elif pgen_type == "PCA4107basis":
        p_gen = PCAGenerator(N_b=4107, approx=True, basis_only=True)
        p_gen.load("pca_gen_4107_imagenet_avg.npy")
    elif pgen_type == "PCA4107basismore":
        p_gen = PCAGenerator(N_b=4107, approx=True, basis_only=True)
        p_gen.load("pca_gen_4107_imagenet_rndavg.npy")
    elif pgen_type == "PCA9408basis":
        p_gen = PCAGenerator(N_b=9408, approx=True, basis_only=True)
        p_gen.load("pca_gen_9408_imagenet_avg.npy")
        # p_gen.load('pca_gen_9408_imagenet_abc.npy')
    elif pgen_type == "PCA9408basismore":
        p_gen = PCAGenerator(N_b=9408, approx=True, basis_only=True)
        p_gen.load("pca_gen_9408_imagenet_rndavg.npy")
        # p_gen.load('pca_gen_9408_imagenet_abc.npy')
    elif pgen_type == "PCA9408basisnormed":
        p_gen = PCAGenerator(N_b=9408, approx=True, basis_only=True)
        p_gen.load("pca_gen_9408_imagenet_normed_avg.npy")
        # p_gen.load('pca_gen_9408_imagenet_abc.npy')
    # elif pgen_type == 'AE9408':
    #     p_gen = AEGenerator(n_channels=3, gpu=args.use_gpu)
    #     p_gen.load_state_dict(torch.load('ae_generator.model'))
    # elif pgen_type == 'GAN128':
    #     p_gen = GANGenerator(n_z=128, n_channels=3, gpu=args.use_gpu)
    #     p_gen.load_state_dict(torch.load('gan128_generator.model'))
    # elif pgen_type == 'VAE9408':
    #     p_gen = VAEGenerator(n_channels=3, gpu=args.use_gpu)
    #     p_gen.load_state_dict(torch.load('vae_generator.model'))
    return p_gen


class ResizeGenerator:
    def __init__(self, factor=4.0):
        self.factor = factor

    def generate_ps(self, inp, N, level=None):
        ps = []
        shape = inp.shape
        assert len(shape) == 3 and shape[0] == 3
        p_small = torch.randn(
            (
                N,
                shape[0],
                int(shape[1] / self.factor),
                int(shape[2] / self.factor),
            ),
            device=inp.device,
        )
        ps = TF.resize(p_small, shape[1:])
        return ps


class DCTGenerator:
    def __init__(self, factor):
        self.factor = factor

    def generate_ps(self, inp, N, level=None):
        C, H, W = inp.shape
        h_use, w_use = int(H / self.factor), int(W / self.factor)
        p_signal = torch.zeros((N,) + inp.shape, device=inp.device)
        # rv = torch.randn((N, C, h_use, w_use), device=inp.device)
        # rv_ortho, _ = torch.linalg.qr(rv, mode='complete')
        rv = torch.randn((N, C, h_use, w_use))
        rv_ortho, _ = torch.linalg.qr(rv, mode="complete")
        rv_ortho = rv_ortho.to(inp.device)
        p_signal[:, :, :h_use, :w_use] = rv_ortho
        im_blocks = blockify(p_signal, H)
        # Orthonormal by default
        dct_blocks = block_idct(im_blocks)
        ps = deblockify(dct_blocks, (H, W))
        return ps
