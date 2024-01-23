"""UNet

Base diffusion model

This is used for MNIST, but could possibly be parameterised to fit both Cifar and MNIST
"""
from pathlib import Path
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange, reduce
from src.model.base import EnergyModel


def load_mnist_diff(diff_path: Path, device, image_size: int = 28, energy: bool = False):
    """Load UNet diffusion model for MNIST"""
    time_emb_dim = 112
    channels = 1
    if energy:
        unet = UNetEnergy(image_size, time_emb_dim, channels)
        unet.load_state_dict(th.load(diff_path))
        unet.to(device)
        unet.train()
    else:
        unet = UNet(image_size, time_emb_dim, channels)
        unet.load_state_dict(th.load(diff_path))
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
        out = super().forward(x, time)
        return ((out - x) ** 2).sum(dim=tuple(i for i in range(1, x.dim())))

    def forward(self, x: th.Tensor, time: th.Tensor):
        energy = self.energy(x, time)
        return th.autograd.grad(energy, x, grad_outputs=th.ones_like(energy), create_graph=True)[0]


def upsample(dim, dim_out):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, dim_out, 3, padding=1),
    )


def downsample(dim, dim_out):
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, dim_out, 1),
    )


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == th.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", th.var)
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=4):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, groups=4):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        time_emb = self.mlp(time_emb)
        time_emb = rearrange(time_emb, "b c -> b c 1 1")
        scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)
        q = q * self.scale

        sim = th.einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = th.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), nn.GroupNorm(1, dim))

    def forward(self, x):
        _, _, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = th.einsum("b h d n, b h e n -> b h d e", k, v)

        out = th.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time: th.Tensor):
        device = time.device
        embeddings = 1.0 / 10000 ** (2.0 / self.dim * th.tensor([i // 2 for i in range(self.dim)], device=device))
        embeddings = time[:, None] * embeddings[None, :]
        embeddings[:, ::2] = embeddings[:, ::2].sin()
        embeddings[:, 1::2] = embeddings[:, 1::2].cos()

        return embeddings
