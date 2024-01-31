"""UNet model for diffusion

Base diffusion model

"""
from typing import Optional
from pathlib import Path
import torch as th
import torch.nn as nn
from src.model.base import EnergyModel
from src.data.cifar import CIFAR_IMAGE_SIZE, CIFAR_NUM_CHANNELS
from src.model.cifar.common import (
    SinusoidalPositionEmbeddings,
    ResnetBlock,
    Attention,
    LinearAttention,
    Residual,
    PreNorm,
    upsample,
    downsample,
)
from src.utils.net import load_params_from_file
from collections import OrderedDict


def load_model(
    model_path: Optional[Path],
    device,
    time_emb_dim: int = 112,
    image_size: int = CIFAR_IMAGE_SIZE,
    num_channels: int = CIFAR_NUM_CHANNELS,
    energy_param: bool = False,
):
    """Load UNET diffusion model from state dict

    The model_path can be a standalone '.th' file or part of a pl checkpoint '.ckpt' file.
    If model_path is None, a new model is initialised.
    """
    if energy_param:
        unet = UNetEnergy(dim=image_size, time_emb_dim=time_emb_dim, channels=num_channels).to(device)
    else:
        unet = UNet(dim=image_size, time_emb_dim=time_emb_dim, channels=num_channels)
    if model_path is not None:
        params = load_params_from_file(model_path)
        if 'ema' in model_path.stem:
            keys = [k for k in params.keys()]
            keys = keys[:int(len(keys) / 2)]
            params_ = OrderedDict()
            for key in keys:
                params_[key] = params[key]
            params = params_
        unet.load_state_dict(params)
    unet.to(device)
    unet.eval()
    return unet


class UNet(nn.Module):
    """UNet
    @param dim: dimension of input x (assumes square (dim x dim) img for now)
    @param time_emb_dim: dimension of time embedding.
    @param channels: number of channels the data have (e.g. 3 for RGB images).
    """

    def __init__(
        self,
        dim: int,
        time_emb_dim: int,
        channels: int,
    ):
        super().__init__()

        self.channels = channels
        self.dim = dim
        self.time_dim = time_emb_dim

        # First & Final Layers
        self.init_conv = nn.Conv2d(channels, dim, 1, padding=0)
        self.final_res_block = ResnetBlock(dim * 2, dim, time_emb_dim=time_emb_dim)
        self.final_conv = nn.Conv2d(dim, channels, 1)

        # Layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        # Time Embedding
        self.time_mlp = nn.Sequential(
            *[
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.GELU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            ]
        )

        # Encoder
        # Block 1
        self.downs.append(
            nn.ModuleList(
                [
                    ResnetBlock(dim, dim, time_emb_dim),
                    ResnetBlock(dim, dim, time_emb_dim),
                    Residual(PreNorm(dim, LinearAttention(dim))),
                    downsample(dim, dim),
                ]
            )
        )
        # Block 2
        self.downs.append(
            nn.ModuleList(
                [
                    ResnetBlock(dim, dim, time_emb_dim),
                    ResnetBlock(dim, dim, time_emb_dim),
                    Residual(PreNorm(dim, LinearAttention(dim))),
                    downsample(dim, 2 * dim),
                ]
            )
        )
        # Block 3
        self.downs.append(
            nn.ModuleList(
                [
                    ResnetBlock(2 * dim, 2 * dim, time_emb_dim),
                    ResnetBlock(2 * dim, 2 * dim, time_emb_dim),
                    Residual(PreNorm(2 * dim, LinearAttention(2 * dim))),
                    nn.Conv2d(2 * dim, 4 * dim, kernel_size=3, padding=1),
                ]
            )
        )

        # Latent Layer
        self.middle = nn.ModuleList(
            [
                ResnetBlock(4 * dim, 4 * dim, time_emb_dim),
                Residual(PreNorm(4 * dim, Attention(4 * dim))),
                ResnetBlock(4 * dim, 4 * dim, time_emb_dim),
            ]
        )

        # Decoder
        # Block 1
        self.ups.append(
            nn.ModuleList(
                [
                    ResnetBlock(6 * dim, 4 * dim, time_emb_dim),
                    ResnetBlock(6 * dim, 4 * dim, time_emb_dim),
                    Residual(PreNorm(4 * dim, LinearAttention(4 * dim))),
                    upsample(4 * dim, 2 * dim),
                ]
            )
        )
        # Block 2
        self.ups.append(
            nn.ModuleList(
                [
                    ResnetBlock(3 * dim, 2 * dim, time_emb_dim),
                    ResnetBlock(3 * dim, 2 * dim, time_emb_dim),
                    Residual(PreNorm(2 * dim, LinearAttention(2 * dim))),
                    upsample(2 * dim, dim),
                ]
            )
        )
        # Block 3
        self.ups.append(
            nn.ModuleList(
                [
                    ResnetBlock(2 * dim, dim, time_emb_dim),
                    ResnetBlock(2 * dim, dim, time_emb_dim),
                    Residual(PreNorm(dim, LinearAttention(dim))),
                    nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                ]
            )
        )

    def forward(self, x: th.Tensor, time: th.Tensor):
        t = self.time_mlp(time)
        return self._unet_forward(x, t)

    def _unet_forward(self, x: th.Tensor, t: th.Tensor):
        x = self.init_conv(x)
        r = x.clone()

        h = []

        # Down-sampling
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        # Latent Space
        for block in self.middle:
            x = block(x, t) if type(block) is ResnetBlock else block(x)

        # Up-sampling
        for block1, block2, attn, upsample in self.ups:
            x = th.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = th.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = th.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


class UNetEnergy(UNet, EnergyModel):
    def __init__(
        self,
        dim: int,
        time_emb_dim: int,
        channels: int,
    ):
        UNet.__init__(self, dim=dim, time_emb_dim=time_emb_dim, channels=channels)
        EnergyModel.__init__(self)

    def energy(self, x: th.Tensor, time: th.Tensor):
        score = super().forward(x, time)
        return ((score - x) ** 2).sum(dim=tuple(i for i in range(1, x.dim())))

    def forward(self, x: th.Tensor, time: th.Tensor):
        energy = self.energy(x, time)
        return th.autograd.grad(energy, x, grad_outputs=th.ones_like(energy), create_graph=True)[0]
