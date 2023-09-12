import torch
import numpy as np
from typing import Callable
from abc import ABC, abstractmethod


class Sampler(ABC):
    def __init__(self, num_samples_per_step: int, step_sizes: torch.tensor, gradient_function: Callable):
        """
        @param num_samples_per_step: Number of MCMC steps per timestep t
        @param step_sizes: Step sizes for each t
        @param gradient_function: Function that returns the score for a given x, t, and text_embedding
        """
        self._step_sizes = step_sizes
        self._num_samples_per_step = num_samples_per_step
        self._gradient_function = gradient_function

    @abstractmethod
    def sample_step(self, *args, **kwargs):
        raise NotImplementedError


class AnnealedULASampler(Sampler):
    """
    Annealed Unadjusted-Langevin Algorithm
    """

    def __init__(self, num_samples_per_step: int, step_sizes: torch.tensor, gradient_function: Callable):
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

    def sample_step(self, x: torch.tensor, t: int, text_embeddings: torch.tensor):
        for i in range(self._num_samples_per_step):
            ss = self._step_sizes[t]
            std = (2 * ss) ** 0.5
            grad = self._gradient_fn_unnorm(x, t, text_embeddings)
            noise = torch.randn_like(x) * std
            x = x + grad * ss + noise
        return x


class AnnealedLAScoreSampler(Sampler):
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
        self._gradient_fn_unnorm = None
        self.n_trapets = n_trapets
        self.accepts = list()
        self.accepts_t = dict()
        self.explogp_t = dict()

    def sample_step(self, x, t, text_embeddings=None):
        print("t", t)

        for i in range(self._num_samples_per_step):
            ss = self._step_sizes[t]
            std = (2 * ss) ** 0.5
            grad = self._gradient_function(x, t, text_embeddings)
            noise = torch.randn_like(x) * std
            mean_x = x + grad * ss
            x_hat = mean_x + noise
            grad_hat = self._gradient_function(x_hat, t, text_embeddings)
            mean_x_hat = x_hat + grad_hat * ss
            # Correction
            logp_reverse = -0.5 * torch.sum((x - mean_x_hat) ** 2) / std**2
            logp_forward = -0.5 * torch.sum((x_hat - mean_x) ** 2) / std**2

            def energy_s(ss_):
                diff = x_hat - x
                x_ = x + ss_[0] * diff
                energy = torch.unsqueeze(torch.sum(self._gradient_fn_unnorm(x_, t, text_embeddings) * diff), dim=0)
                for j in range(1, len(ss_)):
                    x_ = x + ss_[j] * diff
                    energy = torch.concatenate(
                        (
                            energy,
                            torch.unsqueeze(torch.sum(self._gradient_fn_unnorm(x_, t, text_embeddings) * diff), dim=0),
                        )
                    )
                return energy

            s = torch.linspace(0, 1, steps=self.n_trapets).cuda()
            energys = energy_s(s)
            diff_logp_x = torch.trapezoid(energys, s)
            logp_accept = diff_logp_x + logp_reverse - logp_forward
            print("explogp_accept", torch.exp(logp_accept))

            u = torch.rand(1).cuda()
            accept = int(u < torch.exp(logp_accept))
            print("accept", accept)
            self.accepts.append(accept)

            if t not in self.accepts_t.keys():
                self.accepts_t[t] = list()
                self.explogp_t[t] = list()
            self.accepts_t[t].append(accept)
            self.explogp_t[t].append(torch.exp(logp_accept).cpu().numpy())
            print("t_ratio", np.sum(self.accepts_t[t]) / len(self.accepts_t[t]))
            accept = 1.0
            x = accept * x_hat + (1 - accept) * x

        return x


class AnnealedHMCScoreSampler(Sampler):
    """Annealed Metropolis-Hasting Adjusted Hamiltonian Monte Carlo

    Trapezoidal rule is computed with the intermediate steps of HMC (leapfrog steps)
    """

    def __init__(
        self,
        num_samples_per_step: int,
        step_sizes: torch.tensor,
        damping_coeff: float,
        mass_diag_sqrt: torch.tensor,
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
        step_size = self._step_sizes[i]
        return _leapfrog_step(
            x,
            v,
            lambda _x: self._gradient_function(_x, i, text_embeddings),
            step_size,
            self._mass_diag_sqrt[i],
            self._num_leapfrog_steps,
        )

    def sample_step(self, x, t, text_embeddings=None):
        dims = x.dim()

        # Sample Momentum
        v = torch.randn_like(x) * self._mass_diag_sqrt[t]

        for i in range(self._num_samples_per_step):
            # Partial Momentum Refreshment
            eps = torch.randn_like(x)
            v_prime = v * self._damping_coeff + np.sqrt(1.0 - self._damping_coeff**2) * eps * self._mass_diag_sqrt[t]
            x_next, v_next, xs, grads = self.leapfrog_step(x, v_prime, t, text_embeddings)

            logp_v_p = -0.5 * (v_prime**2 / self._mass_diag_sqrt[t] ** 2).sum(dim=tuple(range(1, dims)))
            logp_v = -0.5 * (v_next**2 / self._mass_diag_sqrt[t] ** 2).sum(dim=tuple(range(1, dims)))

            def energy_steps(xs_, grads_):
                e = (grads_[0] * (xs_[1] - xs_[0])).sum(dim=tuple(range(1, dims)))
                e = torch.concatenate((e, (grads_[1] * (xs_[1] - xs_[0])).sum(dim=tuple(range(1, dims)))))
                energy = torch.trapz(e)
                for j in range(1, len(xs_) - 1):
                    e = (grads_[j] * (xs_[j + 1] - xs_[j])).sum(dim=tuple(range(1, dims)))
                    e = torch.concatenate((e, (grads_[j + 1] * (xs_[j + 1] - xs_[j])).sum(dim=tuple(range(1, dims)))))
                    energy += torch.trapz(e)
                return energy

            diff_logp_x = energy_steps(xs, grads)
            logp_accept = logp_v - logp_v_p + diff_logp_x
            u = torch.rand(x_next.shape[0]).to(x_next.device)
            accept = (u < torch.exp(logp_accept)).to(torch.float32)

            # update samples
            x = accept[:, None] * x_next + (1 - accept[:, None]) * x
            v = accept[:, None] * v_next + (1 - accept[:, None]) * v_prime

        return x


def _leapfrog_step(
    x_0: torch.tensor,
    v_0: torch.tensor,
    gradient_target: Callable,
    step_size: float,
    mass_diag_sqrt: torch.tensor,
    num_steps: int,
):
    """Multiple leapfrog steps with"""
    x_k = x_0.clone()
    v_k = v_0.clone()
    if mass_diag_sqrt is None:
        mass_diag_sqrt = torch.ones_like(x_k)

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
