"""Diffusion guidance primitives"""
from abc import ABC, abstractmethod
import torch as th
import torch.nn as nn
from src.diffusion.base import extract, DiffusionSampler


class Guidance(ABC):
    def __init__(self, lambda_):
        self.lambda_ = lambda_

    @abstractmethod
    def grad(self, *args, **kwargs):
        raise NotImplementedError


class GuidanceSampler:
    """Sampling from reconstruction guided DDPM"""

    def __init__(self, diff_model: nn.Module, diff_proc: DiffusionSampler, guidance: Guidance, verbose=False):
        self.diff_model = diff_model
        self.diff_model.eval()
        self.diff_proc = diff_proc
        self.guidance = guidance
        self.verbose = verbose
        self.grads = {"uncond": dict(), "class": dict()}

    @th.no_grad()
    def sample(self, num_samples: int, classes: th.Tensor, device: th.device, shape: tuple):
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

        # self.diff_model.eval()
        steps = []
        classes = classes.to(device)
        x_tm1 = th.randn((num_samples,) + shape).to(device)

        for t in reversed(range(0, self.diff_proc.num_timesteps)):
            if self.verbose and (t + 1) % 100 == 0:
                print(f"Diffusion step {t+1}")
            t_tensor = th.full((x_tm1.shape[0],), t, device=device)
            # Use the model to predict noise and use the noise to step back
            pred_noise = self.diff_model(x_tm1, t_tensor)
            x_tm1 = self._sample_x_tm1_given_x_t(x_tm1, t, pred_noise, classes)
            steps.append(x_tm1.detach().cpu())

        return x_tm1, steps

    def _sample_x_tm1_given_x_t(self, x_t: th.Tensor, t: int, pred_noise: th.Tensor, classes: th.Tensor):
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
        post_var_t = extract(self.diff_proc.posterior_variance, t, x_t)

        if t > 0:
            z = th.randn_like(x_t)
        else:
            z = 0

        sigma_t = self.diff_proc.sigma_t(t, x_t)
        t_tensor = th.full((x_t.shape[0],), t, device=x_t.device)
        if t < 1000:
            class_score = self.guidance.grad(x_t, t_tensor, classes, pred_noise)
        else:
            class_score = th.zeros_like(pred_noise, device=x_t.device)
        self.grads["uncond"][t] = th.norm(-pred_noise)
        self.grads["class"][t] = th.norm(sigma_t * class_score)
        m_tm1 = (x_t + b_t / (th.sqrt(1 - a_bar_t)) * (sigma_t * class_score - pred_noise)) / a_t.sqrt()
        noise = post_var_t.sqrt() * z
        xtm1 = m_tm1 + noise
        return xtm1
