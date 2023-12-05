"""Beta schedules"""
import math
from typing import Tuple
from copy import copy
import torch as th

from src.diffusion.base import compute_alpha_bars


def linear_beta_schedule(beta_start: float = 1e-4, beta_end: float = 0.02, num_timesteps: int = 200) -> th.Tensor:
    """Generate linearly spaced betas

    Args:
        beta_start,
        beta_end,
        num_timesteps

    Returns:
        betas: sequence of betas.
    """
    betas = th.linspace(beta_start, beta_end, num_timesteps)
    return betas


def improved_beta_schedule(num_timesteps: int, s: float = 0.008, beta_max: float = 0.999) -> th.Tensor:
    """Improved beta schedulce

    From A. Nichol and P. Dhariwal (Improved Denoising Diffusion Probabilistic Models)

    Args:
        num_timesteps (corresponds to T in the fomula above),
        s,
        beta_max

    Returns
        beta_t, for all t = 1,...,T as a th.Tensor
    """
    ts = th.arange(1, num_timesteps + 1)
    f_ts = th.cos(th.pi / 2 * (ts / num_timesteps + s) / (1 + s)) ** 2
    f_zeros = math.cos(th.pi / 2 * s / (1 + s)) ** 2
    a_bars_t = f_ts / f_zeros
    a_bars_tm1 = th.cat((th.tensor([1.0]), a_bars_t[:-1]))

    betas = th.min(1 - a_bars_t / a_bars_tm1, th.tensor(beta_max))
    return betas


def respaced_beta_schedule(original_betas: th.Tensor, T: int, respaced_T) -> Tuple[th.Tensor, th.Tensor]:
    if respaced_T == T:
        time_steps = th.arange(T)
        betas = original_betas
    elif respaced_T < T:
        time_steps = respaced_timesteps(T, respaced_T)
        betas = respaced_betas(time_steps, original_betas)
    else:
        raise ValueError("respaced_num_diff_steps cannot be higher than num_diff_steps")

    return betas, time_steps


def respaced_timesteps(num_timesteps: int, desired_num_timesteps: int) -> th.Tensor:
    frac_stride = (num_timesteps - 1) / (desired_num_timesteps - 1)
    time_steps_respace = []
    cur_idx = 0.0
    start_idx = 0
    for _ in range(desired_num_timesteps):
        time_steps_respace.append(start_idx + round(cur_idx))
        cur_idx += frac_stride
    return th.tensor(time_steps_respace)


def respaced_betas(use_timesteps: th.Tensor, original_betas: th.Tensor) -> th.Tensor:
    original_betas = copy(original_betas)
    # timestep_map = []

    alphas_cumprod = compute_alpha_bars((1 - original_betas))
    last_alpha_cumprod = 1.0
    new_betas = []
    for i, alpha_cumprod in enumerate(alphas_cumprod):
        if i in use_timesteps:
            new_beta = 1 - alpha_cumprod / last_alpha_cumprod
            # print(f"a_bar: {alpha_cumprod}, last_a_bar: {last_alpha_cumprod}, new b: {new_beta}")
            new_betas.append(new_beta)
            last_alpha_cumprod = alpha_cumprod
            # timestep_map.append(i)
    return th.tensor(new_betas)


# def sparse_beta_schedule(og_betas: th.Tensor, sparse_factor: int) -> th.Tensor:
#     """Schedule for sparse sampling"""
#     T = og_betas.size(0)
#     alphas = 1 - og_betas
#     new_alphas = th.empty((T // sparse_factor,))
#     for t_sparse, t in enumerate(range(0, T, sparse_factor)):
#         new_alphas[t_sparse] = th.prod(alphas[t : t + sparse_factor])
#         # print(new_alphas[t], alphas[t])
#     return 1 - new_alphas
#
#
# def space_timesteps(num_timesteps, section_counts):
#     """
#     Create a list of timesteps to use from an original diffusion process,
#     given the number of timesteps we want to take from equally-sized portions
#     of the original process.
#
#     For example, if there's 300 timesteps and the section counts are [10,15,20]
#     then the first 100 timesteps are strided to be 10 timesteps, the second 100
#     are strided to be 15 timesteps, and the final 100 are strided to be 20.
#
#     If the stride is a string starting with "ddim", then the fixed striding
#     from the DDIM paper is used, and only one section is allowed.
#
#     :param num_timesteps: the number of diffusion steps in the original
#                           process to divide up.
#     :param section_counts: either a list of numbers, or a string containing
#                            comma-separated numbers, indicating the step count
#                            per section. As a special case, use "ddimN" where N
#                            is a number of steps to use the striding from the
#                            DDIM paper.
#     :return: a set of diffusion steps from the original process to use.
#     """
#     if isinstance(section_counts, str):
#         if section_counts.startswith("ddim"):
#             desired_count = int(section_counts[len("ddim") :])
#             for i in range(1, num_timesteps):
#                 if len(range(0, num_timesteps, i)) == desired_count:
#                     return set(range(0, num_timesteps, i))
#             raise ValueError(f"cannot create exactly {num_timesteps} steps with an integer stride")
#         section_counts = [int(x) for x in section_counts.split(",")]
#     size_per = num_timesteps // len(section_counts)
#     extra = num_timesteps % len(section_counts)
#     start_idx = 0
#     all_steps = []
#     for i, section_count in enumerate(section_counts):
#         size = size_per + (1 if i < extra else 0)
#         if size < section_count:
#             raise ValueError(f"cannot divide section of {size} steps into {section_count}")
#         if section_count <= 1:
#             frac_stride = 1
#         else:
#             frac_stride = (size - 1) / (section_count - 1)
#         cur_idx = 0.0
#         taken_steps = []
#         for _ in range(section_count):
#             taken_steps.append(start_idx + round(cur_idx))
#             cur_idx += frac_stride
#         all_steps += taken_steps
#         start_idx += size
#     return set(all_steps)


if __name__ == "__main__":
    # betas = linear_beta_schedule(num_timesteps=20)
    T = 1000
    respace = 250
    ts = respaced_timesteps(T, respace)
    betas = linear_beta_schedule(num_timesteps=T)
    betas_new = new_sparse(ts, betas)
    print(betas)
    print(betas_new)
