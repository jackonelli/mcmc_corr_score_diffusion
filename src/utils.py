import math
import numpy as np
import matplotlib.pyplot as plt
import chex
import jax


def extract(a: chex.Array, t: chex.Array, x_shape) -> chex.Array:
    """Get coefficients at given timesteps and reshape to [batch_size, 1, ...]."""
    chex.assert_rank(t, 1)
    (bs,) = t.shape
    assert x_shape[0] == bs
    a = jax.device_put(a)
    out = a[t]

    assert out.shape[0] == bs

    return out.reshape([bs] + (len(x_shape) - 1) * [1])


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps, dtype=np.float64)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


def plot_samples(x):
    plt.scatter(x[:, 0], x[:, 1])
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)


def dist_show_2d(fn, xr, yr):
    nticks = 100
    x, y = np.meshgrid(
        np.linspace(xr[0], xr[1], nticks), np.linspace(yr[0], yr[1], nticks)
    )
    coord = np.stack([x, y], axis=-1).reshape((-1, 2))
    heatmap = fn(coord).reshape((nticks, nticks))
    plt.imshow(heatmap)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = np.exp(
        -math.log(max_period) * np.arange(start=0, end=half, type=float) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = np.cat([np.cos(args), np.sin(args)], dim=-1)
    if dim % 2:
        embedding = np.cat([embedding, np.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
