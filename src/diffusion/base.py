"""Diffusion utils"""
from typing import Union
from matplotlib.image import math
import torch as th


def sample_x_t_given_x_0(x_0: th.Tensor, ts: th.Tensor, alphas_bar: th.Tensor):
    """Sample from q(x_t | x_0)

    Add noise to the input tensor x_0 at given timesteps to produce a tensor of noisy samples x_t

    Args:
        x_0: [batch_size * [any shape]]
        ts: [batch_size,]
        alphas_bar [num_timesteps,]

    Returns:
        x_t: batch_size number of samples from x_t ~ q(x_t | x_0) [batch_size, *x.size()]
    """
    a_bar_t = _extract(alphas_bar, ts, x_0)
    noise = th.randn_like(x_0)
    x_t = a_bar_t.sqrt() * x_0 + th.sqrt(1.0 - a_bar_t) * noise

    return x_t


def compute_alpha_bars(alphas):
    """Compute sequence of alpha_bar from sequence of alphas"""
    return th.cumprod(alphas, dim=0)


def _extract(a: th.Tensor, t: Union[int, th.Tensor], x: th.Tensor):
    """Helper function to extract values of a tensor at given time steps

    Args:
        a,
        t,
        x

    Returns:
        out:
    """

    batch_size = x.shape[0]
    out = a.gather(-1, th.full((batch_size,), t) if isinstance(t, int) else t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x.shape) - 1))).to(x.device)
