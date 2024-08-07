"""Diffusion utils"""
from enum import Enum
from abc import ABC
from typing import Union
import torch as th
import torch.nn as nn
import torch.nn.functional as F


# TODO: proper enum for posterior_variance
class PostVar(Enum):
    pass


class DiffusionSampler(ABC):
    """Sampling from DDPM"""

    def __init__(
        self,
        betas: th.Tensor,
        time_steps: th.Tensor,
        posterior_variance="beta",
        energy: bool = False,
    ):
        self.time_steps = time_steps
        self.time_steps_idx = [i for i in range(len(time_steps))]
        self.num_diff_steps = len(time_steps)
        self.verbose_split = list(reversed([i[0].item() for i in th.chunk(self.time_steps, 10)]))

        # define beta
        self.betas = betas
        self.alphas = 1.0 - self.betas
        self.alphas_bar = compute_alpha_bars(self.alphas)

        # Different values of posterior variance (sigma ** 2) proposed of the authors + learned
        if posterior_variance == "beta":
            self.posterior_variance = self.betas
        elif posterior_variance == "beta_tilde":
            self.posterior_variance = (
                self.betas * (1.0 - F.pad(self.alphas_bar[:-1], (1, 0), value=1.0)) / (1.0 - self.alphas_bar)
            )
        elif posterior_variance == "learned":
            self.posterior_variance = "learned"
        else:
            raise NotImplementedError

        self.posterior_log_variance_clipped = _compute_post_log_var(self.betas)

        if energy:
            self.require_grad = True
        else:
            self.require_grad = False

        # current_rng_state = th.get_rng_state()
        # initial_seed = th.initial_seed()
        # th.manual_seed(initial_seed)
        # self.rng_state = th.get_rng_state()
        # th.set_rng_state(current_rng_state)

    def sigma_t(self, t_idx, x_t):
        a_bar_t = extract(self.alphas_bar, t_idx, x_t)
        return th.sqrt(1 - a_bar_t)

    def to(self, device: th.device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_bar = self.alphas_bar.to(device)
        if not isinstance(self.posterior_variance, str):
            self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)

    def sample(self, model: nn.Module, num_samples: int, device: th.device, shape: tuple, verbose=False):
        """Sampling from the backward process
        Sample points from the data distribution

        Args:
            model (model to predict noise)
            num_samples (number of samples)
            device (the device the model is on)
            shape (shape of data, e.g., (1, 28, 28))

        Returns:
            all x through the (predicted) reverse diffusion steps
        """
        if self.require_grad:
            return self._sample_require_grad(model, num_samples, device, shape, verbose)
        else:
            return self._sample(model, num_samples, device, shape, verbose)

    @th.no_grad()
    def _sample(self, model: nn.Module, num_samples: int, device: th.device, shape: tuple, verbose=False):
        steps = []
        x_tm1 = th.randn((num_samples,) + shape).to(device)
        verbose_counter = 0

        for t, t_idx in zip(self.time_steps.__reversed__(), reversed(self.time_steps_idx)):
            t_tensor = th.full((x_tm1.shape[0],), t.item(), device=device)
            t_idx_tensor = th.full((x_tm1.shape[0],), t_idx, device=device)
            if verbose and self.verbose_split[verbose_counter] == t:
                print("Diff step", t.item())
                verbose_counter += 1

            # Use the model to predict noise and use the noise to step back
            if not isinstance(self.posterior_variance, str):
                pred_noise = model(x_tm1, t_tensor)
                sqrt_post_var_t = th.sqrt(extract(self.posterior_variance, t_idx, x_tm1))
                assert pred_noise.size() == x_tm1.size()
            else:
                pred_noise, log_var = model(x_tm1, t_tensor).split(x_tm1.size(1), dim=1)
                log_var, _ = self._clip_var(x_tm1, t_idx_tensor, log_var)
                sqrt_post_var_t = th.exp(0.5 * log_var)
            x_tm1 = self._sample_x_tm1_given_x_t(x_tm1, t_idx, pred_noise, sqrt_post_var_t=sqrt_post_var_t)
            # steps.append((t, x_tm1.detach().cpu()))

        return x_tm1, steps

    def _sample_require_grad(self, model: nn.Module, num_samples: int, device: th.device, shape: tuple, verbose=False):
        steps = []
        x_tm1 = th.randn((num_samples,) + shape).to(device)
        verbose_counter = 0
        # import time
        for t, t_idx in zip(self.time_steps.__reversed__(), reversed(self.time_steps_idx)):
            # start_time = time.time()
            # start = th.cuda.Event(enable_timing=True)
            # end = th.cuda.Event(enable_timing=True)
            # start.record()
            x_tm1 = x_tm1.requires_grad_(True)
            t_tensor = th.full((x_tm1.shape[0],), t.item(), device=device)
            t_idx_tensor = th.full((x_tm1.shape[0],), t_idx, device=device)
            if verbose and self.verbose_split[verbose_counter] == t:
                print("Diff step", t.item())
                verbose_counter += 1

            # Use the model to predict noise and use the noise to step back
            if not isinstance(self.posterior_variance, str):
                pred_noise = model(x_tm1, t_tensor)
                sqrt_post_var_t = th.sqrt(extract(self.posterior_variance, t_idx, x_tm1))
                assert pred_noise.size() == x_tm1.size()
            else:
                pred_noise, log_var = model(x_tm1, t_tensor).split(x_tm1.size(1), dim=1)
                log_var, _ = self._clip_var(x_tm1, t_idx_tensor, log_var)
                sqrt_post_var_t = th.exp(0.5 * log_var)
            pred_noise = pred_noise.detach()
            x_tm1 = self._sample_x_tm1_given_x_t(x_tm1, t_idx, pred_noise, sqrt_post_var_t=sqrt_post_var_t)
            x_tm1 = x_tm1.detach()
            # print(time.time() - start_time)
            # end.record()
            # th.cuda.synchronize()
            # print(start.elapsed_time(end) / 1000)
            # steps.append((t, x_tm1.detach().cpu()))

        return x_tm1, steps

    def _sample_x_tm1_given_x_t(self, x_t: th.Tensor, t: int, pred_noise_t: th.Tensor, sqrt_post_var_t: th.Tensor):
        """Denoise the input tensor at a given timestep using the predicted noise

        Args:
            x_t (any shape),
            t (timestep at which to denoise),
            predicted_noise (noise predicted at the timestep)

        Returns:
            x_tm1 (x[t-1] denoised sample by one step - x_t.shape)
        """

        b_t = extract(self.betas, t, x_t)
        a_t = extract(self.alphas, t, x_t)
        a_bar_t = extract(self.alphas_bar, t, x_t)

        if t > 0:
            # current_rng_state = th.get_rng_state()
            # th.set_rng_state(self.rng_state)
            z = th.randn_like(x_t)
            # self.rng_state = th.get_rng_state()
            # th.set_rng_state(current_rng_state)
        else:
            z = 0

        m_tm1 = (x_t - b_t / (th.sqrt(1 - a_bar_t)) * pred_noise_t) / a_t.sqrt()
        noise = sqrt_post_var_t * z
        xtm1 = m_tm1 + noise
        return xtm1

    def _clip_var(self, x, t, model_var_values):
        min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
        max_log = _extract_into_tensor(th.log(self.betas), t, x.shape)
        # The model_var_values is [-1, 1] for [min_var, max_var].
        frac = (model_var_values + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log
        model_variance = th.exp(model_log_variance)
        return model_log_variance, model_variance

    def q_sample(self, x_0: th.Tensor, ts: th.Tensor, noise: th.Tensor):
        """
        Sampling from the forward process
        Add noise to the input tensor at random timesteps to produce a tensor of noisy samples

        Args:
            x_0 (any shape),
            ts (tensor of timesteps at which to add noise)

        Returns:
            noisy_samples (tensor of noisy samples of shape - x_0.shape)
        """

        x_t = _sample_x_t_given_x_0(x_0, ts, self.alphas_bar, noise)
        return x_t


def _compute_post_log_var(betas):
    alphas = 1.0 - betas
    alphas_bar = compute_alpha_bars(alphas)
    # Shift a_bars: Add 1.0 at t=1, remove val a t=T
    alphas_bar_prev = th.cat(
        (
            th.ones(
                1,
            ),
            alphas_bar[:-1],
        ),
        dim=0,
    )
    # NB: Zero at first entry
    posterior_variance = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
    # Replace zero at t=1, with the val at t=2
    posterior_log_variance_clipped = th.log(
        th.cat(
            (
                posterior_variance[1]
                * th.ones(
                    1,
                ),
                posterior_variance[1:],
            ),
            dim=0,
        )
    )
    return posterior_log_variance_clipped


def _sample_x_t_given_x_0(x_0: th.Tensor, ts: th.Tensor, alphas_bar: th.Tensor, noise: th.Tensor):
    """Sample from q(x_t | x_0)

    Add noise to the input tensor x_0 at given timesteps to produce a tensor of noisy samples x_t

    Args:
        x_0: [batch_size * [any shape]]
        ts: [batch_size,]
        alphas_bar [num_timesteps,]

    Returns:
        x_t: batch_size number of samples from x_t ~ q(x_t | x_0) [batch_size, *x.size()]
    """
    a_bar_t = extract(alphas_bar, ts, x_0)
    x_t = a_bar_t.sqrt() * x_0 + th.sqrt(1.0 - a_bar_t) * noise

    return x_t


def compute_alpha_bars(alphas):
    """Compute sequence of alpha_bar from sequence of alphas"""
    return th.cumprod(alphas, dim=0)


def extract(a: th.Tensor, t: Union[int, th.Tensor], x: th.Tensor):
    """Helper function to extract values of a tensor at given time steps

    Args:
        a,
        t,
        x

    Returns:
        out:
    """

    batch_size = x.shape[0]
    device = a.device
    inds = th.full((batch_size,), t).to(device) if isinstance(t, int) else t.to(device)
    out = a.gather(-1, inds)
    return out.reshape(batch_size, *((1,) * (len(x.shape) - 1))).to(x.device)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extracts indices arr[timesteps], and broadcasts the values to a larger tensor.

    :param arr: the 1-D th.Tensor with T elements.
    :param timesteps: a tensor of indices (diffusion steps) into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = arr.to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
