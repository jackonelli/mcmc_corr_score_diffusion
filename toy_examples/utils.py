import numpy as np
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
