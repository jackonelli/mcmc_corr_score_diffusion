"""High-level sampling functions"""

import torch as th
from src.diffusion.base import compute_alpha_bars


def reverse_diffusion(noise_pred_fn, dim, alpha_ts, sigma_ts, store_traj=False):
    """Reverse diffusion sampling"""
    T = alpha_ts.size(0)
    alpha_bar_ts = compute_alpha_bars(alpha_ts)
    batch_size, img_channels = 1, 1
    x_T = th.randn((batch_size, img_channels, dim, dim))
    x_t = x_T.clone()
    traj = th.empty((T, *x_T.size())) if store_traj else None
    for step, t in enumerate(reversed(range(1, T + 1))):
        print(f"Sampling diffusion step {t}")
        print(x_t.size())
        t_ind = t - 1
        z = th.randn_like(x_T) if t > 1 else th.zeros_like(x_T)
        a_t, a_bar_t, sigma_t = alpha_ts[t_ind], alpha_bar_ts[t_ind], sigma_ts[t_ind]
        t_tensor = th.tensor(t).reshape((1,))
        eps_t = noise_pred_fn(x_t, t_tensor)[0, 0, :, :]
        x_tm1 = _rev_diff_step(x_t, a_t, a_bar_t, eps_t, sigma_t, z)
        if store_traj:
            traj[step] = x_tm1.clone()[0, 0, :, :]
        x_t = x_tm1


def _rev_diff_step(x_t, a_t, a_bar_t, eps_t, sigma_t, z):
    scale = 1 / th.sqrt(a_t)
    noise_factor = (1 - a_t) / th.sqrt(1 - a_bar_t)
    return scale * (x_t - noise_factor * eps_t) + sigma_t * z
