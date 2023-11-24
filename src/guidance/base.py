"""Diffusion guidance primitives"""
from abc import ABC, abstractmethod
import torch as th
import torch.nn as nn
from src.diffusion.base import extract, DiffusionSampler
from src.samplers.mcmc import MCMCSampler


class Guidance(ABC):
    def __init__(self, lambda_):
        self.lambda_ = lambda_

    @abstractmethod
    def grad(self, *args, **kwargs):
        raise NotImplementedError


class GuidanceSampler:
    """Sampling from classifier guided DDPM"""

    def __init__(self, diff_model: nn.Module, diff_proc: DiffusionSampler, guidance: Guidance):
        self.diff_model = diff_model
        self.diff_proc = diff_proc
        self.guidance = guidance
        self.grads = {"uncond": dict(), "class": dict()}

    @th.no_grad()
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
            t_tensor = th.full((x_tm1.shape[0],), t.item(), device=device)
            t_idx_tensor = th.full((x_tm1.shape[0],), t_idx, device=device)
            # Use the model to predict noise and use the noise to step back
            if not isinstance(self.diff_proc.posterior_variance, str):
                pred_noise = self.diff_model(x_tm1, t_tensor)
                sqrt_post_var_t = th.sqrt(extract(self.diff_proc.posterior_variance, t_idx, x_tm1))
                assert pred_noise.size() == x_tm1.size()
            else:
                pred_noise, log_var = self.diff_model(x_tm1, t_tensor).split(x_tm1.size(1), dim=1)
                log_var, _ = self.diff_proc._clip_var(x_tm1, t_idx_tensor, log_var)
                sqrt_post_var_t = th.exp(0.5 * log_var)
            x_tm1 = self._sample_x_tm1_given_x_t(
                x_tm1, t_idx, pred_noise, sqrt_post_var_t=sqrt_post_var_t, classes=classes
            )

            # steps.append(x_tm1.detach().cpu())

        return x_tm1, steps

    def _sample_x_tm1_given_x_t(
        self, x_t: th.Tensor, t: int, pred_noise: th.Tensor, sqrt_post_var_t: th.Tensor, classes: th.Tensor
    ):
        """Denoise the input tensor at a given timestep using the predicted noise

        Args:
            x_t (any shape),
            t (timestep at which to denoise),
            predicted_noise (noise predicted at the timestep)

        Returns:
            x_tm1 (x[t-1] denoised sample by one step - x_t.shape)
        """
        b_t = extract(self.diff_proc.betas, t, x_t)
        a_t = extract(self.diff_proc.alphas, t, x_t)
        a_bar_t = extract(self.diff_proc.alphas_bar, t, x_t)

        if t > 0:
            z = th.randn_like(x_t)
        else:
            z = 0

        sigma_t = self.diff_proc.sigma_t(t, x_t)
        t_tensor = th.full((x_t.shape[0],), t, device=x_t.device)
        class_score = self.guidance.grad(x_t, t_tensor, classes, pred_noise)
        self.grads["uncond"][t] = th.norm(-pred_noise)
        self.grads["class"][t] = th.norm(sigma_t * class_score)
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
        verbose=False,
    ):
        super().__init__(diff_model=diff_model, diff_proc=diff_proc, guidance=guidance, verbose=verbose)
        self.mcmc_sampler = mcmc_sampler
        self.mcmc_sampler.set_gradient_function(self.grad)
        self.reverse = reverse

    def grad(self, x_t, t, classes):
        sigma_t = self.diff_proc.sigma_t(t, x_t)
        t_tensor = th.full((x_t.shape[0],), t, device=x_t.device)
        # pred_noise = self.diff_model(x_t, t_tensor)
        # TODO: Which sigma_t ?
        if not isinstance(self.diff_proc.posterior_variance, str):
            pred_noise = self.diff_model(x_t, t_tensor)
            # sqrt_post_var_t = th.sqrt(extract(self.diff_proc.posterior_variance, t, x_tm1))
            # assert pred_noise.size() == x_t.size()
        else:
            pred_noise, log_var = self.diff_model(x_t, t_tensor).split(x_t.size(1), dim=1)
            # log_var, _ = self.diff_proc._clip_var(x_t, t_tensor, log_var)
            # sqrt_post_var_t = th.exp(0.5 * log_var)
        class_score = self.guidance.grad(x_t, t_tensor, classes, pred_noise)
        return class_score - pred_noise / sigma_t

    @th.no_grad()
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

        self.verbose_counter = 0

        for t, t_idx in zip(self.time_steps.__reversed__(), reversed(self.time_steps_idx)):
            if verbose and self.diff_proc.verbose_split[self.verbose_counter] == t:
                print("Diff step", t.item())
                self.verbose_counter += 1

            if self.reverse:
                t_tensor = th.full((x_tm1.shape[0],), t.item(), device=device)
                t_idx_tensor = th.full((x_tm1.shape[0],), t_idx, device=device)
                # Use the model to predict noise and use the noise to step back
                # pred_noise = self.diff_model(x_tm1, t_tensor)
                if not isinstance(self.diff_proc.posterior_variance, str):
                    pred_noise = self.diff_model(x_tm1, t_tensor)
                    assert pred_noise.size() == x_tm1.size()
                    sqrt_post_var_t = th.sqrt(extract(self.diff_proc.posterior_variance, t_idx, x_tm1))
                else:
                    pred_noise, log_var = self.diff_model(x_tm1, t_tensor).split(x_tm1.size(1), dim=1)
                    assert pred_noise.size() == x_tm1.size()
                    log_var, _ = self.diff_proc._clip_var(x_tm1, t_idx_tensor, log_var)
                    sqrt_post_var_t = th.exp(0.5 * log_var)
                x_tm1 = self._sample_x_tm1_given_x_t(
                    x_tm1, t_idx, pred_noise, sqrt_post_var_t=sqrt_post_var_t, classes=classes
                )

            if t > 0:
                x_tm1 = self.mcmc_sampler.sample_step(x_tm1, t - 1, classes)
            steps.append(x_tm1.detach().cpu())

        return x_tm1, steps
