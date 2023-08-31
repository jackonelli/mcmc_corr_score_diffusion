"""Beta schedules"""
import math
import torch as th


def linear_beta_schedule(beta_start: float = 1e-4, beta_end: float = 0.02, num_timesteps: int = 200):
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


def test_improved_beta_schedule(schedule):
    betas = torch.tensor([0.1012940794, 0.2795438460, 0.4736353534, 0.7240523691, 0.9990000000])
    assert torch.all(torch.isclose(schedule(5), betas))
    print("\033[92m\033[1m✓ Value test passed! \033[0m")
