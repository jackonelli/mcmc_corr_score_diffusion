"""Diffusion guidance primitives"""
from abc import ABC, abstractmethod
import torch as th
import torch.cuda
import torch.nn as nn
import numpy as np
from src.diffusion.base import extract, DiffusionSampler
from src.samplers.mcmc import MCMCSampler
import gc


class Guidance(ABC):
    def __init__(self, lambda_):
        self.lambda_ = lambda_

    @abstractmethod
    def grad(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, *args, **kwargs):
        raise NotImplementedError


class GuidanceSampler:
    """Sampling from classifier guided DDPM"""

    def __init__(self, diff_model: nn.Module, diff_proc: DiffusionSampler, guidance: Guidance, diff_cond: bool = False):
        self.diff_model = diff_model
        self.diff_proc = diff_proc
        self.guidance = guidance
        self.grads = {"uncond": dict(), "class": dict()}
        self.diff_cond = diff_cond

        self.require_g = False
        if 'energy' in dir(self.diff_model):
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

    def _sample_x_tm1_given_x_t(
        self, x_t: th.Tensor, t_idx: int, pred_noise: th.Tensor, sqrt_post_var_t: th.Tensor, classes: th.Tensor
    ):
        """Denoise the input tensor at a given timestep using the predicted noise

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
        class_score = self.guidance.grad(x_t, t_tensor, classes, pred_noise)
        self.grads["uncond"][t_idx] = th.norm(-pred_noise).detach().cpu()
        self.grads["class"][t_idx] = th.norm(sigma_t * class_score).detach().cpu()
        m_tm1 = (x_t + b_t / (th.sqrt(1 - a_bar_t)) * (sigma_t * class_score - pred_noise)) / a_t.sqrt()
        noise = sqrt_post_var_t * z
        xtm1 = m_tm1 + noise
        return xtm1


class MCMCGuidanceSampler(GuidanceSampler):
    def __init__(
        self,
        diff_model: nn.Module,
        diff_proc: DiffusionSampler,
        guidance: Guidance,
        mcmc_sampler: MCMCSampler,
        reverse=True,
        diff_cond: bool = False,
    ):
        super().__init__(diff_model=diff_model, diff_proc=diff_proc, guidance=guidance, diff_cond=diff_cond)
        self.mcmc_sampler = mcmc_sampler
        self.mcmc_sampler.set_gradient_function(self.grad)
        self.reverse = reverse
        self.mcmc_sampler.set_energy_function(self.energy)

    def energy(self, x_t, t, t_idx, classes):
        t_tensor = th.full((x_t.shape[0],), t, device=x_t.device)
        args = [x_t, t_tensor]
        if self.diff_cond:
            args += [classes]
        diff_energy = self.diff_model.energy(*args)
        guidance_energy = self.guidance.log_prob(x_t, t_tensor, classes)
        return guidance_energy + diff_energy

    def grad(self, x_t, t, t_idx, classes):
        """Compute"""
        sigma_t = self.diff_proc.sigma_t(t_idx, x_t)
        t_tensor = th.full((x_t.shape[0],), t, device=x_t.device)
        args = [x_t, t_tensor]
        if self.diff_cond:
            args += [classes]
        if not isinstance(self.diff_proc.posterior_variance, str):
            pred_noise = self.diff_model(*args)
        else:
            pred_noise, _ = self.diff_model(*args).split(x_t.size(1), dim=1)
        class_score = self.guidance.grad(x_t, t_tensor, classes, pred_noise)
        return class_score - pred_noise / sigma_t

    # @th.no_grad()
    def sample(self, num_samples: int, classes: th.Tensor, device: th.device, shape: tuple, verbose=False):
        """Sampling from the backward process
        Sample points from the data distribution

        Args:
            model (model to predict noise)
            num_samples (number of samples)
            classes (num_samples, ): classes to condition on (on for each sample)
            device (the device the model is on)
            shape (shape of data, e.g., (1, 28, 28))

        Returns:
            all x through the (predicted) reverse diffusion steps
        """

        steps = []
        x_tm1 = th.randn((num_samples,) + shape).to(device)

        verbose_counter = 0
        for t, t_idx in zip(self.diff_proc.time_steps.__reversed__(), reversed(self.diff_proc.time_steps_idx)):
            if verbose and self.diff_proc.verbose_split[verbose_counter] == t:
                print("Diff step", t.item())
                verbose_counter += 1

            if self.reverse:
                if self.require_g:
                    x_tm1 = reverse_func_require_grad(self, t, t_idx, x_tm1, classes, device, self.diff_cond)
                else:
                    x_tm1 = reverse_func(self, t, t_idx, x_tm1, classes, device, self.diff_cond)

            if t > 0:
                respaced_t = self.diff_proc.time_steps[t_idx - 1].item()
                x_tm1 = self.mcmc_sampler.sample_step(x_tm1, respaced_t, t_idx - 1, classes)
            x_tm1 = x_tm1.detach()
            # steps.append(x_tm1.detach().cpu())

        return x_tm1, steps


class MCMCGuidanceSamplerStacking(MCMCGuidanceSampler):
    def __init__(
        self,
        diff_model: nn.Module,
        diff_proc: DiffusionSampler,
        guidance: Guidance,
        mcmc_sampler: MCMCSampler,
        reverse: bool = True,
        diff_cond: bool = False,
    ):
        super().__init__(
            diff_model=diff_model,
            diff_proc=diff_proc,
            guidance=guidance,
            mcmc_sampler=mcmc_sampler,
            reverse=reverse,
            diff_cond=diff_cond,
        )

    def sample_stacking(self, num_samples: int, batch_size: int, classes: th.Tensor, device: th.device, shape: tuple):
        n_batches = int(np.ceil(num_samples / batch_size))
        idx = np.array([i * batch_size for i in range(n_batches)] + [num_samples - 1])
        x = th.randn((num_samples,) + shape)

        for t, t_idx in zip(self.diff_proc.time_steps.__reversed__(), reversed(self.diff_proc.time_steps_idx)):
            print("Diff step: ", t.item())
            if self.reverse:
                for i in range(n_batches - 1):
                    x_tm1 = x[idx[i]: idx[i + 1]].to(device)
                    if self.require_g:
                        x_tm1 = reverse_func_require_grad(self, t, t_idx, x_tm1, classes[idx[i]: idx[i + 1]], device,
                                                          self.diff_cond)
                    else:
                        x_tm1 = reverse_func(self, t, t_idx, x_tm1, classes[idx[i]: idx[i + 1]], device, self.diff_cond)

                    x[idx[i]: idx[i + 1]] = x_tm1.detach().cpu()
                    del x_tm1
                    gc.collect()
                    torch.cuda.empty_cache()

            if t > 0:
                # Note x is on cpu!
                respaced_t = self.diff_proc.time_steps[t_idx - 1].item()
                x = self.mcmc_sampler.sample_step(x, respaced_t, t_idx - 1, classes)
        return x, []


@th.no_grad()
def reverse_func(model, t, t_idx, x_tm1, classes, device, diff_cond):
    t_tensor = th.full((x_tm1.shape[0],), t.item(), device=device)
    t_idx_tensor = th.full((x_tm1.shape[0],), t_idx, device=device)
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
