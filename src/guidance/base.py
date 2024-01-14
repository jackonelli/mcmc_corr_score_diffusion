"""Diffusion guidance primitives"""
from abc import ABC, abstractmethod
from typing import Callable
import torch as th
import torch.cuda
import torch.nn as nn
import numpy as np
from src.diffusion.base import extract, DiffusionSampler
from src.samplers.mcmc import (
    MCMCSampler,
    MCMCMHCorrSampler,
    AnnealedLAScoreSampler,
    AnnealedLAEnergySampler,
    langevin_step,
    get_mean,
    transition_factor,
    estimate_energy_diff_linear,
)
from src.model.base import EnergyModel
from typing import List
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
        if isinstance(self.diff_model, EnergyModel):
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

        full_trajs = []
        x_tm1 = th.randn((num_samples,) + shape).to(device)
        full_trajs.append(x_tm1.detach().cpu())
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
            full_trajs.append(x_tm1.detach().cpu())

        return x_tm1, full_trajs

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
        mcmc_sampling_predicate: Callable = lambda t: t > 0,
        reverse=True,
        diff_cond: bool = False,
    ):
        super().__init__(diff_model=diff_model, diff_proc=diff_proc, guidance=guidance, diff_cond=diff_cond)
        self.mcmc_sampler = mcmc_sampler
        # Function which maps diff step t to a bool, controlling for which timesteps to to MCMC sampling.
        self._mcmc_sampling_predicate = mcmc_sampling_predicate
        self.mcmc_sampler.set_gradient_function(self.grad)
        self.reverse = reverse
        self.mcmc_sampler.set_energy_function(self.energy)

    def energy(self, x_t, t, t_idx, classes):
        sigma_t = self.diff_proc.sigma_t(t_idx, x_t).squeeze()
        t_tensor = th.full((x_t.shape[0],), t, device=x_t.device)
        args = [x_t, t_tensor]
        if self.diff_cond:
            args += [classes]
        diff_energy = self.diff_model.energy(*args)
        guidance_energy = self.guidance.log_prob(x_t, t_tensor, classes)
        return guidance_energy - diff_energy / sigma_t

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

    def grad_energy(self, x_t, t, t_idx, classes):
        sigma_t = self.diff_proc.sigma_t(t_idx, x_t)
        t_tensor = th.full((x_t.shape[0],), t, device=x_t.device)
        args = [x_t, t_tensor]
        if self.diff_cond:
            args += [classes]
        if not isinstance(self.diff_proc.posterior_variance, str):
            energy_grad = self.diff_model(*args)
        else:
            energy_grad, _ = self.diff_model(*args).split(x_t.size(1), dim=1)
        class_score = self.guidance.grad(x_t, t_tensor, classes, None)
        return class_score - energy_grad / sigma_t

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

        full_trajs = []
        x_tm1 = th.randn((num_samples,) + shape).to(device)
        full_trajs.append(x_tm1.detach().cpu())

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

            if self._mcmc_sampling_predicate(t):
                respaced_t = self.diff_proc.time_steps[t_idx - 1].item()
                x_tm1 = self.mcmc_sampler.sample_step(x_tm1, respaced_t, t_idx - 1, classes)
            x_tm1 = x_tm1.detach()
            full_trajs.append(x_tm1.detach().cpu())

        return x_tm1, full_trajs


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
                    x_tm1 = x[idx[i] : idx[i + 1]].to(device)
                    if self.require_g:
                        x_tm1 = reverse_func_require_grad(
                            self, t, t_idx, x_tm1, classes[idx[i] : idx[i + 1]], device, self.diff_cond
                        )
                    else:
                        x_tm1 = reverse_func(
                            self, t, t_idx, x_tm1, classes[idx[i] : idx[i + 1]], device, self.diff_cond
                        )

                    x[idx[i] : idx[i + 1]] = x_tm1.detach().cpu()
                    del x_tm1
                    gc.collect()
                    torch.cuda.empty_cache()

            if t > 0:
                # Note x is on cpu!
                respaced_t = self.diff_proc.time_steps[t_idx - 1].item()
                x = self.mcmc_sampler.sample_step(x, respaced_t, t_idx - 1, classes)
        return x, []


class GuidanceSamplerAcceptanceComparison:
    def __init__(self, guidance_models: List[MCMCGuidanceSampler]):
        self.guidance_models = guidance_models
        self.n_models = len(guidance_models)

    def accept_ratio_trajectory(
        self,
        num_samples: int,
        classes: th.Tensor,
        device: th.device,
        shape: tuple,
        i_model: int = 0,
        n_per_t: int = 1,
        seed: int = 0,
        verbose=False,
    ):
        """Given a model generate reverse trajectories and at each timestep MCMC sample

        Args:
            num_samples (number of samples)
            classes (num_samples, ): classes to condition on (on for each sample)
            device (the device the model is on)
            shape (shape of data, e.g., (1, 28, 28))
            i_model (which model to use for reverse)
            n_per_t (number of samples per timestep for estimating the acceptance ratio)
            seed (same seed for each model)

        Returns:
            acceptance ratios for each model at

        """

        assert all([isinstance(self.guidance_models[i].mcmc_sampler, MCMCMHCorrSampler) for i in range(self.n_models)])
        x_tm1 = th.randn((num_samples,) + shape).to(device)
        acceptance = {
            i: {j.item(): list() for j in self.guidance_models[i].diff_proc.time_steps} for i in range(self.n_models)
        }
        energy_diff = {
            i: {j.item(): list() for j in self.guidance_models[i].diff_proc.time_steps} for i in range(self.n_models)
        }

        verbose_counter = 0
        self_ = self.guidance_models[i_model]
        for t, t_idx in zip(self_.diff_proc.time_steps.__reversed__(), reversed(self_.diff_proc.time_steps_idx)):
            if verbose and self_.diff_proc.verbose_split[verbose_counter] == t:
                print("Diff step", t.item())
                verbose_counter += 1

            if self_.reverse:
                if self_.require_g:
                    x_tm1 = reverse_func_require_grad(self_, t, t_idx, x_tm1, classes, device, self_.diff_cond)
                else:
                    x_tm1 = reverse_func(self_, t, t_idx, x_tm1, classes, device, self_.diff_cond)

            if t > 0:
                current_rng_state = th.get_rng_state()
                respaced_t = self_.diff_proc.time_steps[t_idx - 1].item()
                for i in range(self.n_models):
                    th.manual_seed(seed + t)
                    for j in range(n_per_t):
                        _ = self.guidance_models[i].mcmc_sampler.sample_step(x_tm1, respaced_t, t_idx - 1, classes)
                        acceptance[i][t.item() - 1].append(self.guidance_models[i].mcmc_sampler.all_accepts[respaced_t])
                        energy_diff[i][t.item() - 1].append(
                            self.guidance_models[i].mcmc_sampler.energy_diff[respaced_t]
                        )
                th.set_rng_state(current_rng_state)
            x_tm1 = x_tm1.detach()

        return acceptance, energy_diff

    def accept_ratio_one_guides_LA(
        self, num_samples: int, classes: th.Tensor, device: th.device, shape: tuple, seed: int = 0, verbose=False
    ):
        """Given a model generate reverse trajectories and one model guides and compare same points with other

        Args:
            num_samples (number of samples)
            classes (num_samples, ): classes to condition on (on for each sample)
            device (the device the model is on)
            shape (shape of data, e.g., (1, 28, 28))
            seed (same seed for each model)

        Returns:
            acceptance ratios for each model at

        """

        assert all(
            [
                isinstance(self.guidance_models[i].mcmc_sampler, AnnealedLAEnergySampler)
                or isinstance(self.guidance_models[i].mcmc_sampler, AnnealedLAScoreSampler)
                for i in range(self.n_models)
            ]
        )

        acceptance = {
            i_: {
                i: {j.item(): list() for j in self.guidance_models[i].diff_proc.time_steps}
                for i in range(self.n_models)
            }
            for i_ in range(self.n_models)
        }
        energy_d = {
            i_: {
                i: {j.item(): list() for j in self.guidance_models[i].diff_proc.time_steps}
                for i in range(self.n_models)
            }
            for i_ in range(self.n_models)
        }

        for i_model in range(self.n_models):
            verbose_counter = 0
            x_tm1 = th.randn((num_samples,) + shape).to(device)
            self_ = self.guidance_models[i_model]
            for t, t_idx in zip(self_.diff_proc.time_steps.__reversed__(), reversed(self_.diff_proc.time_steps_idx)):
                if verbose and self_.diff_proc.verbose_split[verbose_counter] == t:
                    print("Diff step", t.item())
                    verbose_counter += 1

                if self_.reverse:
                    if self_.require_g:
                        x_tm1 = reverse_func_require_grad(self_, t, t_idx, x_tm1, classes, device, self_.diff_cond)
                    else:
                        x_tm1 = reverse_func(self_, t, t_idx, x_tm1, classes, device, self_.diff_cond)

                if t > 0:
                    current_rng_state = th.get_rng_state()
                    t_idx = t_idx - 1
                    t = self_.diff_proc.time_steps[t_idx].item()

                    dims = x_tm1.dim()
                    for step in range(self_.mcmc_sampler.num_samples_per_step):
                        th.manual_seed(seed + t + step)

                        x_tm1.requires_grad_(True)
                        x_hat, mean_x, ss = langevin_step(
                            x_tm1,
                            t,
                            t_idx,
                            classes,
                            self_.mcmc_sampler.step_sizes,
                            self_.mcmc_sampler.gradient_function,
                        )
                        x_hat.requires_grad_(True)
                        mean_x_hat = get_mean(self_.mcmc_sampler.gradient_function, x_hat, t, t_idx, ss, classes)
                        logp_reverse, logp_forward = transition_factor(x_tm1, mean_x, x_hat, mean_x_hat, ss)

                        if isinstance(self_.mcmc_sampler, AnnealedLAEnergySampler):
                            energy_diff = self_.mcmc_sampler.energy_function(
                                x_hat, t, t_idx, classes
                            ) - self_.mcmc_sampler.energy_function(x_tm1, t, t_idx, classes)
                        else:
                            n_trapets = self_.mcmc_sampler.n_trapets
                            intermediate_steps = th.linspace(0, 1, steps=n_trapets).to(x_tm1.device)
                            energy_diff = estimate_energy_diff_linear(
                                self_.mcmc_sampler.gradient_function,
                                x_tm1,
                                x_hat,
                                t,
                                t_idx,
                                intermediate_steps,
                                classes,
                                dims,
                            )
                        logp_accept = energy_diff + logp_reverse - logp_forward
                        acceptance[i_model][i_model][t].append(th.exp(logp_accept).detach().cpu())
                        energy_d[i_model][i_model][t].append(energy_diff.detach().cpu())
                        u = th.rand(x_tm1.shape[0]).to(x_tm1.device)
                        accept = (
                            (u < th.exp(logp_accept))
                            .to(th.float32)
                            .reshape((x_tm1.shape[0],) + tuple(([1 for _ in range(dims - 1)])))
                        )
                        idxs = [j for j, _ in enumerate(self.guidance_models) if j != i_model]
                        models = [j_m for j, j_m in enumerate(self.guidance_models) if j != i_model]
                        for model_j, j in zip(models, idxs):
                            ss = model_j.mcmc_sampler.step_sizes[t]
                            mean_x = get_mean(model_j.mcmc_sampler.gradient_function, x_tm1, t, t_idx, ss, classes)
                            mean_x_hat = get_mean(model_j.mcmc_sampler.gradient_function, x_hat, t, t_idx, ss, classes)
                            logp_reverse, logp_forward = transition_factor(x_tm1, mean_x, x_hat, mean_x_hat, ss)
                            if isinstance(model_j.mcmc_sampler, AnnealedLAEnergySampler):
                                energy_diff = model_j.mcmc_sampler.energy_function(
                                    x_hat, t, t_idx, classes
                                ) - model_j.mcmc_sampler.energy_function(x_tm1, t, t_idx, classes)
                            else:
                                n_trapets = model_j.mcmc_sampler.n_trapets
                                intermediate_steps = th.linspace(0, 1, steps=n_trapets).to(x_tm1.device)
                                energy_diff = estimate_energy_diff_linear(
                                    model_j.mcmc_sampler.gradient_function,
                                    x_tm1,
                                    x_hat,
                                    t,
                                    t_idx,
                                    intermediate_steps,
                                    classes,
                                    dims,
                                )
                            logp_accept = energy_diff + logp_reverse - logp_forward
                            acceptance[i_model][j][t].append(th.exp(logp_accept).detach().cpu())
                            energy_d[i_model][j][t].append(energy_diff.detach().cpu())
                        x_tm1 = accept * x_hat + (1 - accept) * x_tm1
                    th.set_rng_state(current_rng_state)
                x_tm1 = x_tm1.detach()

        return acceptance, energy_d


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
