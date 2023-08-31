"""High-level sampling functions"""

import torch as th
from src.diffusion.base import compute_alpha_bars


def reverse_diffusion(noise_pred_fn, dim, alpha_ts, sigma_ts, store_traj=False):
    """Reverse diffusion sampling"""
    with th.no_grad():
        T = alpha_ts.size(0)
        alpha_bar_ts = compute_alpha_bars(alpha_ts)
        x_T = th.randn((dim, dim))
        x_t = x_T.clone()
        traj = th.empty((T, *x_T.size())) if store_traj else None
        for step, t in enumerate(reversed(range(1, T + 1))):
            z = th.randn_like(x_T) if t > 1 else th.zeros_like(x_T)
            a_t, a_bar_t, sigma_t = alpha_ts[t], alpha_bar_ts[t], sigma_ts[t]
            eps_t = noise_pred_fn(x_t, t)
            x_tm1 = _rev_diff_step(x_t, a_t, a_bar_t, eps_t, sigma_t, z)
            if store_traj:
                traj[step] = x_tm1.clone()
            x_t = x_tm1


def _rev_diff_step(x_t, a_t, a_bar_t, eps_t, sigma_t, z):
    scale = 1 / th.sqrt(a_t)
    noise_factor = (1 - a_t) / th.sqrt(1 - a_bar_t)
    return scale * (x_t - noise_factor * eps_t) + sigma_t * z
