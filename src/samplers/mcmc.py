import torch
import torch as th
import numpy as np
from typing import Callable
import itertools
from abc import ABC, abstractmethod
import gc


class MCMCSampler(ABC):
    def __init__(self, num_samples_per_step: int, step_sizes: th.Tensor, gradient_function: Callable):
        """
        @param num_samples_per_step: Number of MCMC steps per timestep t
        @param step_sizes: Step sizes for each t
        @param gradient_function: Function that returns the score for a given x, t, and text_embedding
        """
        self.step_sizes = step_sizes
        self.num_samples_per_step = num_samples_per_step
        self.gradient_function = gradient_function
        self.accept_ratio = dict()
        self.all_accepts = dict()

    @abstractmethod
    def sample_step(self, *args, **kwargs):
        raise NotImplementedError

    def set_gradient_function(self, gradient_function):
        self.gradient_function = gradient_function


class AnnealedULASampler(MCMCSampler):
    """
    Annealed Unadjusted-Langevin Algorithm
    """

    def __init__(self, num_samples_per_step: int, step_sizes: th.Tensor, gradient_function: Callable):
        """
        @param num_samples_per_step: Number of ULA steps per timestep t
        @param step_sizes: Step sizes for each t
        @param gradient_function: Function that returns the score for a given x, t, and text_embedding
        """
        super().__init__(
            num_samples_per_step=num_samples_per_step, step_sizes=step_sizes, gradient_function=gradient_function
        )
        self._sync_function = None
        self._noise_function = None
        self._gradient_fn_unnorm = None

    @th.no_grad()
    def sample_step(self, x: th.Tensor, t: int, text_embeddings: th.Tensor):
        for i in range(self.num_samples_per_step):
            ss = self.step_sizes[t]
            std = (2 * ss) ** 0.5
            grad = self._gradient_fn_unnorm(x, t, text_embeddings)
            noise = th.randn_like(x) * std
            x = x + grad * ss + noise
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
        self._sync_function = None
        self._noise_function = None
        self.n_trapets = n_trapets

    @th.no_grad()
    def sample_step(self, x, t, text_embeddings=None):
        dims = x.dim()
        print("t", t)
        self.accept_ratio[t] = list()
        self.all_accepts[t] = list()

        for i in range(self.num_samples_per_step):
            ss = self.step_sizes[t]
            std = (2 * ss) ** 0.5
            grad = self.gradient_function(x, t, text_embeddings)
            noise = th.randn_like(x) * std
            mean_x = x + grad * ss
            x_hat = mean_x + noise
            grad_hat = self.gradient_function(x_hat, t, text_embeddings)
            mean_x_hat = x_hat + grad_hat * ss
            # Correction
            logp_reverse = -0.5 * th.sum((x - mean_x_hat) ** 2) / std**2
            logp_forward = -0.5 * th.sum((x_hat - mean_x) ** 2) / std**2

            def energy_s(ss_):
                diff = x_hat - x
                x_ = x + ss_[0] * diff
                energy = th.unsqueeze(th.sum(self.gradient_function(x_, t, text_embeddings) * diff), dim=0)
                for j in range(1, len(ss_)):
                    x_ = x + ss_[j] * diff
                    energy = th.concatenate(
                        (
                            energy,
                            th.unsqueeze(th.sum(self.gradient_function(x_, t, text_embeddings) * diff), dim=0),
                        )
                    )
                return energy

            s = th.linspace(0, 1, steps=self.n_trapets).cuda()
            energys = energy_s(s)
            diff_logp_x = th.trapezoid(energys, s)
            logp_accept = diff_logp_x + logp_reverse - logp_forward

            u = th.rand(x.shape[0]).to(x.device)
            accept = (
                (u < th.exp(logp_accept)).to(th.float32).reshape((x.shape[0],) + tuple(([1 for _ in range(dims - 1)])))
            )
            self.accept_ratio[t].append((th.sum(accept) / accept.shape[0]).detach().cpu().item())
            self.all_accepts[t].append(accept.detach().cpu())
            x = accept * x_hat + (1 - accept) * x

        return x


class AnnealedHMCScoreSampler(MCMCSampler):
    """Annealed Metropolis-Hasting Adjusted Hamiltonian Monte Carlo

    Trapezoidal rule is computed with the intermediate steps of HMC (leapfrog steps)
    """

    def __init__(
        self,
        num_samples_per_step: int,
        step_sizes: th.Tensor,
        damping_coeff: float,
        mass_diag_sqrt: th.Tensor,
        num_leapfrog_steps: int,
        gradient_function: Callable,
    ):
        """
        @param num_samples_per_step: Number of HMC steps per timestep t
        @param step_sizes: Step sizes for leapfrog steps for each t
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
        self._sync_function = None

    def leapfrog_step(self, x, v, i, text_embeddings):
        step_size = self.step_sizes[i]
        return _leapfrog_step(
            x,
            v,
            lambda _x: self.gradient_function(_x, i, text_embeddings),
            step_size,
            self._mass_diag_sqrt[i],
            self._num_leapfrog_steps,
        )

    @th.no_grad()
    def sample_step(self, x, t, text_embeddings=None):
        dims = x.dim()

        # Sample Momentum
        v = th.randn_like(x) * self._mass_diag_sqrt[t]
        self.accept_ratio[t] = list()
        self.all_accepts[t] = list()

        for i in range(self.num_samples_per_step):
            # Partial Momentum Refreshment
            eps = th.randn_like(x)
            v_prime = v * self._damping_coeff + np.sqrt(1.0 - self._damping_coeff**2) * eps * self._mass_diag_sqrt[t]
            x_next, v_next, xs, grads = self.leapfrog_step(x, v_prime, t, text_embeddings)

            logp_v_p = -0.5 * (v_prime**2 / self._mass_diag_sqrt[t] ** 2).sum(dim=tuple(range(1, dims)))
            logp_v = -0.5 * (v_next**2 / self._mass_diag_sqrt[t] ** 2).sum(dim=tuple(range(1, dims)))

            def energy_steps(xs_, grads_):
                e = (grads_[0] * (xs_[1] - xs_[0])).sum(dim=tuple(range(1, dims))).reshape(-1, 1)
                e = th.cat((e, (grads_[1] * (xs_[1] - xs_[0])).sum(dim=tuple(range(1, dims))).reshape(-1, 1)), 1)
                energy = th.trapz(e)
                for j in range(1, len(xs_) - 1):
                    e = (grads_[j] * (xs_[j + 1] - xs_[j])).sum(dim=tuple(range(1, dims))).reshape(-1, 1)
                    e = th.cat(
                        (e, (grads_[j + 1] * (xs_[j + 1] - xs_[j])).sum(dim=tuple(range(1, dims))).reshape(-1, 1)), 1
                    )
                    energy += th.trapz(e)
                return energy

            diff_logp_x = energy_steps(xs, grads)
            logp_accept = logp_v - logp_v_p + diff_logp_x
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


def _leapfrog_step(
    x_0: th.Tensor,
    v_0: th.Tensor,
    gradient_target: Callable,
    step_size: float,
    mass_diag_sqrt: th.Tensor,
    num_steps: int,
):
    """Multiple leapfrog steps with"""
    x_k = x_0.clone()
    v_k = v_0.clone()
    if mass_diag_sqrt is None:
        mass_diag_sqrt = th.ones_like(x_k)

    mass_diag = mass_diag_sqrt**2.0
    xs, grads = list(), list()
    xs.append(x_k)
    grad = gradient_target(x_k)
    grads.append(grad.clone())

    for _ in range(num_steps):  # Inefficient version - should combine half steps
        v_k += 0.5 * step_size * grad  # half step in v
        x_k += step_size * v_k / mass_diag  # Step in x
        xs.append(x_k.clone())
        grad = gradient_target(x_k)
        grads.append(grad.clone())
        v_k += 0.5 * step_size * grad  # half step in v
    return x_k, v_k, xs, grads


class AdaptiveStepSizeMCMCSamplerWrapper(MCMCSampler):
    def __init__(self, sampler: MCMCSampler, accept_rate_bound: list, max_iter: int = 10):
        super().__init__(
            num_samples_per_step=sampler.num_samples_per_step,
            step_sizes=sampler.step_sizes,
            gradient_function=sampler.gradient_function,
        )
        self.sampler = sampler
        self.accept_rate_bound = accept_rate_bound
        self.max_iter = max_iter
        self.T = self.sampler.step_sizes.shape[0]
        self.res = {i: {"accepts": [], "step_sizes": []} for i in range(self.T - 1)}

    def sample_step(self, x, t, text_embeddings=None):
        state = torch.get_rng_state()
        step_found = False
        upper = {"stepsize": None, "accept": 1}
        lower = {"stepsize": None, "accept": 0}
        if t < self.T - 2:
            self.sampler.step_sizes[t] = self.res[t + 1]["step_sizes"][-1]

        i = 0
        x_ = None
        while not step_found and i < self.max_iter:
            torch.manual_seed(t)
            x_ = self.sampler.sample_step(x, t, text_embeddings)
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
        return x_

    def set_gradient_function(self, gradient_function):
        self.sampler.set_gradient_function(gradient_function)


class AdaptiveStepSizeMCMCSamplerWrapperSmallBatchSize(MCMCSampler):
    def __init__(self, sampler: MCMCSampler, accept_rate_bound: list, batch_size: int, device, max_iter: int = 10):
        super().__init__(
            num_samples_per_step=sampler.num_samples_per_step,
            step_sizes=sampler.step_sizes,
            gradient_function=sampler.gradient_function,
        )
        self.sampler = sampler
        self.accept_rate_bound = accept_rate_bound
        self.max_iter = max_iter
        self.T = self.sampler.step_sizes.shape[0]
        self.res = {i: {"accepts": [], "step_sizes": []} for i in range(self.T - 1)}
        self.batch_size = batch_size
        self.device = device

    def sample_step(self, x, t, text_embeddings=None):
        state = torch.get_rng_state()
        step_found = False
        upper = {"stepsize": None, "accept": 1}
        lower = {"stepsize": None, "accept": 0}
        if t < self.T - 2:
            self.sampler.step_sizes[t] = self.res[t + 1]["step_sizes"][-1]

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
                x_ = self.sampler.sample_step(x_, t, y_)
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
