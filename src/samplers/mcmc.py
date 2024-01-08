import torch
import torch as th
import numpy as np
from typing import Callable, Dict, Optional
import itertools
from abc import ABC, abstractmethod
import gc


class MCMCSampler(ABC):
    def __init__(
        self,
        num_samples_per_step: int,
        step_sizes: Dict[int, float],
        gradient_function: Callable,
        energy_function: Optional[Callable] = None,
    ):
        """
        @param num_samples_per_step: Number of MCMC steps per timestep t
        @param step_sizes: Step sizes for each t
        @param gradient_function: Function that returns the score for a given x, t, and text_embedding
        @param energy_function: Function that returns the energy for a given x, t, and text_embedding
        """
        self.step_sizes = step_sizes
        self.num_samples_per_step = num_samples_per_step
        self.gradient_function = gradient_function
        self.energy_function = energy_function
        self.accept_ratio = dict()
        self.all_accepts = dict()

    @abstractmethod
    def sample_step(self, *args, **kwargs):
        raise NotImplementedError

    def set_gradient_function(self, gradient_function):
        self.gradient_function = gradient_function

    def set_energy_function(self, energy_function):
        self.energy_function = energy_function


class AnnealedULAEnergySampler(MCMCSampler):
    """
    Annealed Unadjusted-Langevin Algorithm
    """

    def __init__(
        self,
        num_samples_per_step: int,
        step_sizes: Dict[int, float],
        gradient_function: Callable,
        energy_function: Optional[Callable] = None,
    ):
        """
        @param num_samples_per_step: Number of ULA steps per timestep t
        @param step_sizes: Step sizes for each t
        @param gradient_function: Function that returns the score for a given x, t, and text_embedding
        @param energy_function: Function that returns the energy for a given x, t, and text_embedding
        """
        super().__init__(
            num_samples_per_step=num_samples_per_step,
            step_sizes=step_sizes,
            gradient_function=gradient_function,
            energy_function=energy_function,
        )

    # NB: t_idx not in use.
    def sample_step(self, x: th.Tensor, t: int, t_idx: int, classes: th.Tensor):
        for i in range(self.num_samples_per_step):
            x.requires_grad_(True)
            x, _, _ = langevin_step(x, t, t_idx, classes, self.step_sizes, self.gradient_function)
            x = x.detach()
        return x


class AnnealedULAScoreSampler(MCMCSampler):
    """
    Annealed Unadjusted-Langevin Algorithm
    """

    def __init__(self, num_samples_per_step: int, step_sizes: Dict[int, float], gradient_function: Callable):
        """
        @param num_samples_per_step: Number of ULA steps per timestep t
        @param step_sizes: Step sizes for each t
        @param gradient_function: Function that returns the score for a given x, t, and text_embedding
        """
        super().__init__(
            num_samples_per_step=num_samples_per_step, step_sizes=step_sizes, gradient_function=gradient_function
        )

    @th.no_grad()
    # NB: t_idx not in use.
    def sample_step(self, x: th.Tensor, t: int, t_idx: int, classes: th.Tensor):
        for i in range(self.num_samples_per_step):
            x, _, _ = langevin_step(x, t, t_idx, classes, self.step_sizes, self.gradient_function)
        return x


class AnnealedLAScoreSampler(MCMCSampler):
    """Annealed Metropolis-Hasting Adjusted Langevin Algorithm

    A straight line is used to compute the trapezoidal rule
    """

    def __init__(self, num_samples_per_step: int, step_sizes, gradient_function, n_trapets=5):
        """
        @param num_samples_per_step: Number of LA steps per timestep t
        @param step_sizes: Step sizes for each t
        @param gradient_function: Function that returns the score for a given x, t, and text_embedding
        @param n_trapets: Number of steps used for the trapezoidal rule
        """
        super().__init__(
            num_samples_per_step=num_samples_per_step, step_sizes=step_sizes, gradient_function=gradient_function
        )
        self.n_trapets = n_trapets

    @th.no_grad()
    def sample_step(self, x, t, t_idx, classes=None):
        dims = x.dim()
        self.accept_ratio[t] = list()
        self.all_accepts[t] = list()

        for i in range(self.num_samples_per_step):
            x_hat, mean_x, ss = langevin_step(x, t, t_idx, classes, self.step_sizes, self.gradient_function)

            mean_x_hat = get_mean(self.gradient_function, x_hat, t, t_idx, ss, classes)
            logp_reverse, logp_forward = transition_factor(x, mean_x, x_hat, mean_x_hat, ss)

            intermediate_steps = th.linspace(0, 1, steps=self.n_trapets).to(x.device)
            energy_diff = estimate_energy_diff_linear(
                self.gradient_function, x, x_hat, t, t_idx, intermediate_steps, classes, dims
            )
            logp_accept = energy_diff + logp_reverse - logp_forward

            u = th.rand(x.shape[0]).to(x.device)
            accept = (
                (u < th.exp(logp_accept)).to(th.float32).reshape((x.shape[0],) + tuple(([1 for _ in range(dims - 1)])))
            )
            self.accept_ratio[t].append((th.sum(accept) / accept.shape[0]).detach().cpu().item())
            self.all_accepts[t].append(accept.detach().cpu())
            x = accept * x_hat + (1 - accept) * x

        return x


class AnnealedLAEnergySampler(MCMCSampler):
    """Annealed Metropolis-Hasting Adjusted Langevin Algorithm using energy parameterization"""

    def __init__(self, num_samples_per_step: int, step_sizes, gradient_function, energy_function=None):
        """
        @param num_samples_per_step: Number of LA steps per timestep t
        @param step_sizes: Step sizes for each t
        @param gradient_function: Function that returns the score for a given x, t, and text_embedding
        @param energy_function: Function that returns the energy for a given x, t, and text_embedding
        """
        super().__init__(
            num_samples_per_step=num_samples_per_step,
            step_sizes=step_sizes,
            gradient_function=gradient_function,
            energy_function=energy_function,
        )

    def sample_step(self, x, t, t_idx, classes=None):
        dims = x.dim()
        self.accept_ratio[t] = list()
        self.all_accepts[t] = list()

        for i in range(self.num_samples_per_step):
            x.requires_grad_(True)
            x_hat, mean_x, ss = langevin_step(x, t, t_idx, classes, self.step_sizes, self.gradient_function)
            x_hat.requires_grad_(True)

            mean_x_hat = get_mean(self.gradient_function, x_hat, t, t_idx, ss, classes)
            logp_reverse, logp_forward = transition_factor(x, mean_x, x_hat, mean_x_hat, ss)

            energy_diff = self.energy_function(x_hat, t, t_idx, classes) - self.energy_function(x, t, t_idx, classes)
            logp_accept = energy_diff + logp_reverse - logp_forward

            u = th.rand(x.shape[0]).to(x.device)
            accept = (
                (u < th.exp(logp_accept)).to(th.float32).reshape((x.shape[0],) + tuple(([1 for _ in range(dims - 1)])))
            )
            self.accept_ratio[t].append((th.sum(accept) / accept.shape[0]).detach().cpu().item())
            self.all_accepts[t].append(accept.detach().cpu())
            x = accept * x_hat + (1 - accept) * x
            x = x.detach()
        return x


def langevin_step(x, t, t_idx, classes, step_sizes, gradient_function):
    ss = step_sizes[t]
    std = (2 * ss) ** 0.5
    mean_x = get_mean(gradient_function, x, t, t_idx, ss, classes)
    noise = th.randn_like(x) * std
    x_hat = mean_x + noise
    return x_hat, mean_x, ss


def get_mean(gradient_function, x, t, t_idx, ss, classes):
    """Get mean of transition distribution"""
    grad = gradient_function(x, t, t_idx, classes)
    return x + grad * ss


def transition_factor(x, mean_x, x_hat, mean_x_hat, ss):
    std = (2 * ss) ** 0.5
    logp_reverse = -0.5 * th.sum((x - mean_x_hat) ** 2) / std**2
    logp_forward = -0.5 * th.sum((x_hat - mean_x) ** 2) / std**2
    return logp_reverse, logp_forward


def estimate_energy_diff_linear(gradient_function, x, x_hat, t, t_idx, ss_, classes, dims):
    diff = x_hat - x
    x_ = x + ss_[0] * diff
    e = (gradient_function(x_, t, t_idx, classes) * diff).sum(dim=tuple(range(1, dims))).reshape(-1, 1)
    for j in range(1, len(ss_)):
        x_ = x + ss_[j] * diff
        e = th.cat(
            (e, (gradient_function(x_, t, t_idx, classes) * diff).sum(dim=tuple(range(1, dims))).reshape(-1, 1)), 1
        )
    return th.trapz(e, ss_)


class AnnealedUHMCScoreSampler(MCMCSampler):
    """Annealed Unadjusted Hamiltonian Monte Carlo using score parameterization"""

    def __init__(
        self,
        num_samples_per_step: int,
        step_sizes: Dict[int, float],
        damping_coeff: float,
        mass_diag_sqrt: th.Tensor,
        num_leapfrog_steps: int,
        gradient_function: Callable,
    ):
        """
        @param num_samples_per_step: Number of HMC steps per timestep t
        @param step_sizes: Step size for leapfrog steps for each t
        @param damping_coeff: Damping coefficient
        @param mass_diag_sqrt: Square root of mass diagonal matrix for each t
        @param num_leapfrog_steps: Number of leapfrog steps per HMC step
        @param gradient_function: Function that returns the score for a given x, t, and text_embedding
        """
        super().__init__(
            num_samples_per_step=num_samples_per_step, step_sizes=step_sizes, gradient_function=gradient_function
        )
        self._damping_coeff = damping_coeff
        self._mass_diag_sqrt = mass_diag_sqrt
        self._num_leapfrog_steps = num_leapfrog_steps

    @th.no_grad()
    def sample_step(self, x, t, t_idx, classes=None):
        # Sample Momentum
        v = th.randn_like(x) * self._mass_diag_sqrt[t_idx]
        self.accept_ratio[t] = list()
        self.all_accepts[t] = list()

        for _ in range(self.num_samples_per_step):
            # Partial Momentum Refreshment
            v_prime = get_v_prime(v=v, damping_coeff=self._damping_coeff, mass_diag_sqrt=self._mass_diag_sqrt[t_idx])

            x, v, _, _ = leapfrog_steps(
                x_0=x,
                v_0=v_prime,
                t=t,
                t_idx=t_idx,
                gradient_function=self.gradient_function,
                step_size=self.step_sizes[t],
                mass_diag_sqrt=self._mass_diag_sqrt[t_idx],
                num_steps=self._num_leapfrog_steps,
                classes=classes,
            )
        return x


class AnnealedUHMCEnergySampler(MCMCSampler):
    """Annealed Unadjusted Hamiltonian Monte Carlo using energy-parameterization"""

    def __init__(
        self,
        num_samples_per_step: int,
        step_sizes: Dict[int, float],
        damping_coeff: float,
        mass_diag_sqrt: th.Tensor,
        num_leapfrog_steps: int,
        gradient_function: Callable,
        energy_function: Optional[Callable] = None,
    ):
        """
        @param num_samples_per_step: Number of HMC steps per timestep t
        @param step_sizes: Step size for leapfrog steps for each t
        @param damping_coeff: Damping coefficient
        @param mass_diag_sqrt: Square root of mass diagonal matrix for each t
        @param num_leapfrog_steps: Number of leapfrog steps per HMC step
        @param gradient_function: Function that returns the score for a given x, t, and text_embedding
        @param energy_function: Function that returns the energy for a given x, t, and text_embedding
        """
        super().__init__(
            num_samples_per_step=num_samples_per_step,
            step_sizes=step_sizes,
            gradient_function=gradient_function,
            energy_function=energy_function,
        )
        self._damping_coeff = damping_coeff
        self._mass_diag_sqrt = mass_diag_sqrt
        self._num_leapfrog_steps = num_leapfrog_steps

    def sample_step(self, x, t, t_idx, classes=None):
        # Sample Momentum
        v = th.randn_like(x) * self._mass_diag_sqrt[t_idx]
        self.accept_ratio[t] = list()
        self.all_accepts[t] = list()

        for _ in range(self.num_samples_per_step):
            # Partial Momentum Refreshment
            v_prime = get_v_prime(v=v, damping_coeff=self._damping_coeff, mass_diag_sqrt=self._mass_diag_sqrt[t_idx])

            x, v, _, _ = leapfrog_steps(
                x_0=x,
                v_0=v_prime,
                t=t,
                t_idx=t_idx,
                gradient_function=self.gradient_function,
                step_size=self.step_sizes[t],
                mass_diag_sqrt=self._mass_diag_sqrt[t_idx],
                num_steps=self._num_leapfrog_steps,
                classes=classes,
            )
            x = x.detach()
        return x


class AnnealedHMCScoreSampler(MCMCSampler):
    """Annealed Metropolis-Hasting Adjusted Hamiltonian Monte Carlo

    Trapezoidal rule is computed with the intermediate steps of HMC (leapfrog steps)
    """

    def __init__(
        self,
        num_samples_per_step: int,
        step_sizes: Dict[int, float],
        damping_coeff: float,
        mass_diag_sqrt: th.Tensor,
        num_leapfrog_steps: int,
        gradient_function: Callable,
    ):
        """
        @param num_samples_per_step: Number of HMC steps per timestep t
        @param step_sizes: Step size for leapfrog steps for each t
        @param damping_coeff: Damping coefficient
        @param mass_diag_sqrt: Square root of mass diagonal matrix for each t
        @param num_leapfrog_steps: Number of leapfrog steps per HMC step
        @param gradient_function: Function that returns the score for a given x, t, and text_embedding
        """
        super().__init__(
            num_samples_per_step=num_samples_per_step, step_sizes=step_sizes, gradient_function=gradient_function
        )
        self._damping_coeff = damping_coeff
        self._mass_diag_sqrt = mass_diag_sqrt
        self._num_leapfrog_steps = num_leapfrog_steps

    @th.no_grad()
    def sample_step(self, x, t, t_idx, classes=None):
        dims = x.dim()

        # Sample Momentum
        v = th.randn_like(x) * self._mass_diag_sqrt[t_idx]
        self.accept_ratio[t] = list()
        self.all_accepts[t] = list()

        for _ in range(self.num_samples_per_step):
            # Partial Momentum Refreshment
            v_prime = get_v_prime(v=v, damping_coeff=self._damping_coeff, mass_diag_sqrt=self._mass_diag_sqrt[t_idx])

            x_next, v_next, xs, grads = leapfrog_steps(
                x_0=x,
                v_0=v_prime,
                t=t,
                t_idx=t_idx,
                gradient_function=self.gradient_function,
                step_size=self.step_sizes[t],
                mass_diag_sqrt=self._mass_diag_sqrt[t_idx],
                num_steps=self._num_leapfrog_steps,
                classes=classes,
            )

            logp_v_p, logp_v = transition_hmc(
                v_prime=v_prime, v_next=v_next, mass_diag_sqrt=self._mass_diag_sqrt[t_idx], dims=dims
            )

            # Energy diff estimation
            energy_diff = estimate_energy_diff(xs, grads, dims)
            logp_accept = logp_v - logp_v_p + energy_diff

            u = th.rand(x_next.shape[0]).to(x_next.device)
            accept = (
                (u < th.exp(logp_accept))
                .to(th.float32)
                .reshape((x_next.shape[0],) + tuple(([1 for _ in range(dims - 1)])))
            )

            # update samples
            x = accept * x_next + (1 - accept) * x
            v = accept * v_next + (1 - accept) * v_prime
            self.accept_ratio[t].append((th.sum(accept) / accept.shape[0]).detach().cpu().item())
            self.all_accepts[t].append(accept.detach().cpu().squeeze())
        return x


class AnnealedHMCEnergySampler(MCMCSampler):
    """Annealed Metropolis-Hasting Adjusted Hamiltonian Monte Carlo using energy-parameterization"""

    def __init__(
        self,
        num_samples_per_step: int,
        step_sizes: Dict[int, float],
        damping_coeff: float,
        mass_diag_sqrt: th.Tensor,
        num_leapfrog_steps: int,
        gradient_function: Callable,
        energy_function: Optional[Callable] = None,
    ):
        """
        @param num_samples_per_step: Number of HMC steps per timestep t
        @param step_sizes: Step size for leapfrog steps for each t
        @param damping_coeff: Damping coefficient
        @param mass_diag_sqrt: Square root of mass diagonal matrix for each t
        @param num_leapfrog_steps: Number of leapfrog steps per HMC step
        @param gradient_function: Function that returns the score for a given x, t, and text_embedding
        @param energy_function: Function that returns the energy for a given x, t, and text_embedding
        """
        super().__init__(
            num_samples_per_step=num_samples_per_step,
            step_sizes=step_sizes,
            gradient_function=gradient_function,
            energy_function=energy_function,
        )
        self._damping_coeff = damping_coeff
        self._mass_diag_sqrt = mass_diag_sqrt
        self._num_leapfrog_steps = num_leapfrog_steps

    def sample_step(self, x, t, t_idx, classes=None):
        dims = x.dim()

        # Sample Momentum
        v = th.randn_like(x) * self._mass_diag_sqrt[t_idx]
        self.accept_ratio[t] = list()
        self.all_accepts[t] = list()

        for i in range(self.num_samples_per_step):
            # Partial Momentum Refreshment
            v_prime = get_v_prime(v=v, damping_coeff=self._damping_coeff, mass_diag_sqrt=self._mass_diag_sqrt[t_idx])

            x_next, v_next, _, _ = leapfrog_steps(
                x_0=x,
                v_0=v_prime,
                t=t,
                t_idx=t_idx,
                gradient_function=self.gradient_function,
                step_size=self.step_sizes[t],
                mass_diag_sqrt=self._mass_diag_sqrt[t_idx],
                num_steps=self._num_leapfrog_steps,
                classes=classes,
            )

            logp_v_p, logp_v = transition_hmc(
                v_prime=v_prime, v_next=v_next, mass_diag_sqrt=self._mass_diag_sqrt[t_idx], dims=dims
            )

            # Energy diff estimation
            energy_diff = self.energy_function(x_next, t, t_idx, classes) - self.energy_function(x, t, t_idx, classes)
            logp_accept = logp_v - logp_v_p + energy_diff

            u = th.rand(x_next.shape[0]).to(x_next.device)
            accept = (
                (u < th.exp(logp_accept))
                .to(th.float32)
                .reshape((x_next.shape[0],) + tuple(([1 for _ in range(dims - 1)])))
            )

            # update samples
            x = accept * x_next + (1 - accept) * x
            x = x.detach()
            v = accept * v_next + (1 - accept) * v_prime
            self.accept_ratio[t].append((th.sum(accept) / accept.shape[0]).detach().cpu().item())
            self.all_accepts[t].append(accept.detach().cpu().squeeze())
        return x


def leapfrog_steps(
    x_0: th.Tensor,
    v_0: th.Tensor,
    t: int,
    t_idx: int,
    gradient_function: Callable,
    step_size: float,
    mass_diag_sqrt: th.Tensor,
    num_steps: int,
    classes: Optional[th.Tensor],
):
    """Multiple leapfrog steps with"""
    x_k = x_0.clone()
    v_k = v_0.clone()
    if mass_diag_sqrt is None:
        mass_diag_sqrt = th.ones_like(x_k)

    mass_diag = mass_diag_sqrt**2.0
    xs, grads = list(), list()
    xs.append(x_k)
    x_k = x_k.requires_grad_(True)
    grad = gradient_function(x_k, t, t_idx, classes).detach()
    grads.append(grad.clone())

    for _ in range(num_steps):  # Inefficient version - should combine half steps
        v_k += 0.5 * step_size * grad  # half step in v
        x_k = (x_k + step_size * v_k / mass_diag).detach()  # Step in x
        xs.append(x_k.clone())
        x_k = x_k.requires_grad_(True)
        grad = gradient_function(x_k, t, t_idx, classes)
        grads.append(grad.clone().detach())
        v_k += 0.5 * step_size * grad  # half step in v
    return x_k, v_k, xs, grads


def get_v_prime(v, damping_coeff, mass_diag_sqrt):
    eps = th.randn_like(v)
    v_prime = v * damping_coeff + np.sqrt(1.0 - damping_coeff**2) * eps * mass_diag_sqrt
    return v_prime


def transition_hmc(v_prime, v_next, mass_diag_sqrt, dims):
    logp_v_p = -0.5 * (v_prime**2 / mass_diag_sqrt**2).sum(dim=tuple(range(1, dims)))
    logp_v = -0.5 * (v_next**2 / mass_diag_sqrt**2).sum(dim=tuple(range(1, dims)))
    return logp_v_p, logp_v


def estimate_energy_diff(xs_, grads_, dims):
    e = (grads_[0] * (xs_[1] - xs_[0])).sum(dim=tuple(range(1, dims))).reshape(-1, 1)
    e = th.cat((e, (grads_[1] * (xs_[1] - xs_[0])).sum(dim=tuple(range(1, dims))).reshape(-1, 1)), 1)
    energy = th.trapz(e)
    for j in range(1, len(xs_) - 1):
        e = (grads_[j] * (xs_[j + 1] - xs_[j])).sum(dim=tuple(range(1, dims))).reshape(-1, 1)
        e = th.cat((e, (grads_[j + 1] * (xs_[j + 1] - xs_[j])).sum(dim=tuple(range(1, dims))).reshape(-1, 1)), 1)
        energy += th.trapz(e)
    return energy


class AdaptiveStepSizeMCMCSamplerWrapper(MCMCSampler):
    def __init__(self, sampler: MCMCSampler, accept_rate_bound: list, time_steps, max_iter: int = 10):
        super().__init__(
            num_samples_per_step=sampler.num_samples_per_step,
            step_sizes=sampler.step_sizes,
            gradient_function=sampler.gradient_function,
        )
        self.sampler = sampler
        self.accept_rate_bound = accept_rate_bound
        self.max_iter = max_iter
        self.respaced_T = time_steps.size(0)
        self.time_steps = time_steps
        self.res = {t.item(): {"accepts": [], "step_sizes": []} for t in time_steps}

    def sample_step(self, x, t, t_idx, classes=None):
        state = torch.get_rng_state()
        step_found = False
        upper = {"stepsize": None, "accept": 1}
        lower = {"stepsize": None, "accept": 0}
        if t_idx < self.respaced_T - 2:
            prev_respaced_t = self.time_steps[t_idx + 1].item()
            self.sampler.step_sizes[t] = th.tensor(self.res[prev_respaced_t]["step_sizes"][-1])

        i = 0
        x_ = None
        while not step_found and i < self.max_iter:
            torch.manual_seed(t)
            x_ = self.sampler.sample_step(x, t, t_idx, classes)
            x_ = x_.detach()
            a_rate = np.mean(self.sampler.accept_ratio[t])
            step_s = self.sampler.step_sizes[t].clone()
            self.res[t]["accepts"].append(a_rate)
            self.res[t]["step_sizes"].append(step_s.detach().cpu().item())
            if self.accept_rate_bound[0] <= a_rate <= self.accept_rate_bound[1]:
                step_found = True
            else:
                # Update best bound so far
                if self.accept_rate_bound[1] < a_rate <= upper["accept"]:
                    upper["accept"] = a_rate
                    upper["stepsize"] = step_s
                if lower["accept"] <= a_rate < self.accept_rate_bound[0]:
                    lower["accept"] = a_rate
                    lower["stepsize"] = step_s

                # New step size
                step_s_ = step_s.clone()
                if upper["stepsize"] is not None and lower["stepsize"] is not None:
                    suggest_new = torch.exp((torch.log(upper["stepsize"]) + torch.log(lower["stepsize"])) / 2)
                    if suggest_new == step_s_:
                        w = th.rand(1).item()
                        new_step_s = torch.exp(
                            w * torch.log(upper["stepsize"]) + (1 - w) * torch.log(lower["stepsize"])
                        )
                    else:
                        new_step_s = suggest_new
                else:
                    if a_rate > self.accept_rate_bound[1]:
                        new_step_s = step_s_ / 10
                    else:  # a_rate < self.accept_rate_bound[0]
                        new_step_s = step_s_ * 10

                self.sampler.step_sizes[t] = new_step_s
            i += 1
        torch.set_rng_state(state)
        return x_

    def set_gradient_function(self, gradient_function):
        self.sampler.set_gradient_function(gradient_function)


class AdaptiveStepSizeMCMCSamplerWrapperSmallBatchSize(MCMCSampler):
    def __init__(
        self, sampler: MCMCSampler, accept_rate_bound: list, time_steps, batch_size: int, device, max_iter: int = 10
    ):
        super().__init__(
            num_samples_per_step=sampler.num_samples_per_step,
            step_sizes=sampler.step_sizes,
            gradient_function=sampler.gradient_function,
        )
        self.sampler = sampler
        self.accept_rate_bound = accept_rate_bound
        self.max_iter = max_iter
        self.respaced_T = time_steps.size(0)
        self.time_steps = time_steps
        self.res = {t.item(): {"accepts": [], "step_sizes": []} for t in time_steps}
        self.batch_size = batch_size
        self.device = device

    def sample_step(self, x, t, t_idx, text_embeddings=None):
        state = torch.get_rng_state()
        step_found = False
        upper = {"stepsize": None, "accept": 1}
        lower = {"stepsize": None, "accept": 0}
        if t < self.respaced_T - 2:
            prev_respaced_t = self.time_steps[t_idx + 1].item()
            self.sampler.step_sizes[t] = th.tensor(self.res[prev_respaced_t]["step_sizes"][-1])

        i = 0
        n_batches = int(np.ceil(x.shape[0] / self.batch_size))
        idx = np.array([i * self.batch_size for i in range(n_batches)] + [x.shape[0] - 1])
        x_next = torch.empty(x.size())

        while not step_found and i < self.max_iter:
            torch.manual_seed(t)
            accepts = list()
            for j in range(n_batches - 1):
                x_ = x[idx[j] : idx[j + 1]].to(self.device)
                y_ = text_embeddings[idx[j] : idx[j + 1]].to(self.device)
                x_ = self.sampler.sample_step(x_, t, t_idx, y_)
                x_next[idx[j] : idx[j + 1]] = x_.detach().cpu()
                accepts += list(itertools.chain(*[acc.numpy().tolist() for acc in self.sampler.all_accepts[t]]))
                del x_
                del y_
                gc.collect()
                torch.cuda.empty_cache()
            a_rate = np.mean(accepts)
            step_s = self.sampler.step_sizes[t].clone()
            self.res[t]["accepts"].append(a_rate)
            self.res[t]["step_sizes"].append(step_s.detach().cpu().item())
            if self.accept_rate_bound[0] <= a_rate <= self.accept_rate_bound[1]:
                step_found = True
            else:
                # Update best bound so far
                if self.accept_rate_bound[1] < a_rate <= upper["accept"]:
                    upper["accept"] = a_rate
                    upper["stepsize"] = step_s
                if lower["accept"] <= a_rate < self.accept_rate_bound[0]:
                    lower["accept"] = a_rate
                    lower["stepsize"] = step_s

                # New step size
                new_step_s = step_s.clone()
                if upper["stepsize"] is None:
                    new_step_s /= 10
                elif lower["stepsize"] is None:
                    new_step_s *= 10
                else:
                    new_step_s = torch.exp((torch.log(upper["stepsize"]) + torch.log(lower["stepsize"])) / 2)

                self.sampler.step_sizes[t] = new_step_s
            i += 1
        torch.set_rng_state(state)
        return x_next

    def set_gradient_function(self, gradient_function):
        self.sampler.set_gradient_function(gradient_function)

    def set_energy_function(self, energy_function):
        self.sampler.set_energy_function(energy_function)
