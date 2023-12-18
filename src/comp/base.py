"""Diffusion guidance primitives"""
from abc import ABC, abstractmethod
import torch as th
import torch.cuda
import torch.nn as nn
import numpy as np
from src.diffusion.base import extract, DiffusionSampler
from src.samplers.mcmc import MCMCSampler
import gc


class ProductComp(ABC):
    def __init__(self, lambda_):
        self.lambda_ = lambda_

    @abstractmethod
    def grad(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, *args, **kwargs):
        raise NotImplementedError


class ProductCompSampler:
    """Sampling from product distribution of diffusion models"""

    def __init__(
        self, diff_model_1: nn.Module, diff_model_2: nn.Module, diff_proc: DiffusionSampler, diff_cond: bool = False
    ):
        self.diff_model_1 = diff_model_1
        self.diff_model_2 = diff_model_2
        self.diff_proc = diff_proc
        self.guidance = guidance
        self.grads = {"diff_1": dict(), "diff_2": dict()}
        self.diff_cond = diff_cond

        self.require_g = False
        if "energy" in dir(self.diff_model_1) or "energy" in dir(self.diff_model_2):
            self.require_g = True

        # Seed only for noise in reverse step
        current_rng_state = th.get_rng_state()
        initial_seed = th.initial_seed()
        th.manual_seed(initial_seed)
        self.rng_state = th.get_rng_state()
        th.set_rng_state(current_rng_state)

    def sample(self, num_samples: int, classes: th.Tensor, device: th.device, shape: tuple, verbose=False):
        """Sample points from the data distribution by running the reverse process

        Args:
            num_samples (number of samples)
            classes (num_samples, ): classes to condition on (on for each sample)
            device (the device the model is on)
            shape (shape of data, e.g., (1, 28, 28))

        Returns:
            all x through the (predicted) reverse diffusion steps
        """

        steps = []
        x_tm1 = th.randn((num_samples,) + shape).to(device)
        self.verbose_counter = 0

        for t, t_idx in zip(self.diff_proc.time_steps.__reversed__(), reversed(self.diff_proc.time_steps_idx)):
            if verbose and self.diff_proc.verbose_split[self.verbose_counter] == t:
                print("Diff step", t.item())
                self.verbose_counter += 1
            if self.require_g:
                x_tm1 = reverse_func_require_grad(self, t, t_idx, x_tm1, classes, device, self.diff_cond)
            else:
                x_tm1 = reverse_func(self, t, t_idx, x_tm1, classes, device, self.diff_cond)
            x_tm1 = x_tm1.detach()
            # steps.append(x_tm1.detach().cpu())

        return x_tm1, steps

    def _sample_x_tm1_given_x_t(self, x_t: th.Tensor, t_idx: int, pred_noise: th.Tensor, sqrt_post_var_t: th.Tensor):
        """Denoise the input tensor at a given timestep using the predicted noise based on a product distribution.

        Args:
            x_t (any shape),
            t_idx (timestep index at which to denoise),
            predicted_noise (noise predicted at the timestep)

        Returns:
            x_tm1 (x[t-1] denoised sample by one step - x_t.shape)
        """
        b_t = extract(self.diff_proc.betas, t_idx, x_t)
        a_t = extract(self.diff_proc.alphas, t_idx, x_t)
        a_bar_t = extract(self.diff_proc.alphas_bar, t_idx, x_t)

        if t_idx > 0:
            current_rng_state = th.get_rng_state()
            th.set_rng_state(self.rng_state)
            z = th.randn_like(x_t)
            self.rng_state = th.get_rng_state()
            th.set_rng_state(current_rng_state)
        else:
            z = 0

        sigma_t = self.diff_proc.sigma_t(t_idx, x_t)
        t_tensor = th.full((x_t.shape[0],), t_idx, device=x_t.device)
        # self.grads["diff_1"][t_idx] = th.norm(-pred_noise).detach().cpu()
        m_tm1 = (x_t + b_t / (th.sqrt(1 - a_bar_t)) * -pred_noise) / a_t.sqrt()
        noise = sqrt_post_var_t * z
        xtm1 = m_tm1 + noise
        return xtm1


@th.no_grad()
def reverse_func_prod(model, t, t_idx, x_tm1, device):
    t_tensor = th.full((x_tm1.shape[0],), t.item(), device=device)
    t_idx_tensor = th.full((x_tm1.shape[0],), t_idx, device=device)
    # Use the model to predict noise and use the noise to step back
    args = [x_tm1, t_tensor]
    if not isinstance(model.diff_proc.posterior_variance, str):
        pred_noise_1 = model.diff_model_1(*args)
        assert pred_noise_1.size() == x_tm1.size()
        pred_noise_2 = model.diff_model_2(*args)
        assert pred_noise_2.size() == x_tm1.size()
        # Note, common posterior variance for the product components
        sqrt_post_var_t = th.sqrt(extract(model.diff_proc.posterior_variance, t_idx, x_tm1))
        pred_noise = pred_noise_1 + pred_noise_2
    else:
        raise NotImplemented("Learned variance for product composition, not implemented.")
        pred_noise, log_var = model.diff_model(*args).split(x_tm1.size(1), dim=1)
        assert pred_noise.size() == x_tm1.size()
        log_var, _ = model.diff_proc._clip_var(x_tm1, t_idx_tensor, log_var)
        sqrt_post_var_t = th.exp(0.5 * log_var)
    x_tm1 = model._sample_x_tm1_given_x_t(x_tm1, t_idx, pred_noise, sqrt_post_var_t=sqrt_post_var_t)
    return x_tm1


def reverse_func_require_grad(model, t, t_idx, x_tm1, classes, device, diff_cond):
    t_tensor = th.full((x_tm1.shape[0],), t.item(), device=device)
    t_idx_tensor = th.full((x_tm1.shape[0],), t_idx, device=device)
    x_tm1 = x_tm1.requires_grad_(True)
    # Use the model to predict noise and use the noise to step back
    args = [x_tm1, t_tensor]
    if diff_cond:
        args += [classes.to(device)]
    if not isinstance(model.diff_proc.posterior_variance, str):
        pred_noise = model.diff_model(*args)
        assert pred_noise.size() == x_tm1.size()
        sqrt_post_var_t = th.sqrt(extract(model.diff_proc.posterior_variance, t_idx, x_tm1))
    else:
        pred_noise, log_var = model.diff_model(*args).split(x_tm1.size(1), dim=1)
        assert pred_noise.size() == x_tm1.size()
        log_var, _ = model.diff_proc._clip_var(x_tm1, t_idx_tensor, log_var)
        sqrt_post_var_t = th.exp(0.5 * log_var)
    x_tm1 = model._sample_x_tm1_given_x_t(x_tm1, t_idx, pred_noise, sqrt_post_var_t=sqrt_post_var_t, classes=classes)
    return x_tm1
