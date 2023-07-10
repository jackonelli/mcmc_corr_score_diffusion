import jax
import jax.numpy as jnp
import haiku as hk
import distrax
import chex
import numpy as np
import optax
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import tensorflow_datasets as tfds
import functools
import pandas as pd
from functools import partial
from jax import jvp
import jax
import jax.numpy as jnp
import distrax
import haiku as hk
# pylint: disable=g-bare-generic
from typing import Callable, Optional, Tuple, Union, Dict
Array = jnp.ndarray
Scalar = Union[float, int]
RandomKey = Array

GradientTarget = Callable[[Array, Array], Array]
InitialSampler = Callable[[Array], Array]


class AnnealedULASampler:
    """Implements AIS with ULA"""

    def __init__(self,
                 num_steps: int,
                 num_samples_per_step: int,
                 step_sizes: jnp.array,
                 initial_distribution: distrax.Distribution,
                 target_distribution,
                 gradient_function):
        assert len(step_sizes) == num_steps, "Must have as many stepsizes as intermediate distributions."
        self._step_sizes = step_sizes
        self._num_steps = num_steps
        self._num_samples_per_step = num_samples_per_step
        self._initial_distribution = initial_distribution
        self._target_distribution = target_distribution
        if target_distribution is None:
            assert gradient_function is not None
            self._gradient_function = gradient_function
        else:
            self._gradient_function = jax.grad(
                lambda x, i: target_distribution(x, i).sum())

        self._total_steps = self._num_samples_per_step * (self._num_steps - 1)

    def transition_distribution(self, i, x):
        ss = self._step_sizes[i]
        std = (2 * ss) ** .5
        grad = self._gradient_function(x, i)
        mu = x + grad * ss
        dist = distrax.MultivariateNormalDiag(mu, jnp.ones_like(mu) * std)
        return dist

    def sample(self, key: RandomKey, n_samples: int):
        init_key, key = jax.random.split(key)
        x = self._initial_distribution.sample(seed=init_key,
                                              sample_shape=(n_samples,))
        logw = -self._initial_distribution.log_prob(x)

        inputs = (key, logw, x)

        def body_fn(i, inputs):
            key, logw, x = inputs
            dist_ind = (i // self._num_samples_per_step)
            dist_forward = self.transition_distribution(dist_ind, x)
            sample_key, key = jax.random.split(key)
            x_hat = dist_forward.sample(seed=sample_key)
            dist_reverse = self.transition_distribution(dist_ind - 1, x_hat)
            logw += dist_reverse.log_prob(x) - dist_forward.log_prob(x_hat)
            x = x_hat
            return key, logw, x

        _, logw, x = jax.lax.fori_loop(0, self._total_steps, body_fn, inputs)

        if self._target_distribution is not None:
            logw += self._target_distribution(x, self._num_steps - 1)
        else:
            logw = None

        return x, logw, None

    def logp_raise(self, key: RandomKey, x: jnp.array):
        logw = jnp.zeros((x.shape[0],))

        inputs = (key, logw, x)

        def body_fn(i, inputs):
            key, logw, x = inputs
            ind = i // self._num_samples_per_step
            dist_ind = self._num_steps - 1 - ind
            dist_reverse = self.transition_distribution(dist_ind - 1, x)
            sample_key, key = jax.random.split(key)
            x_hat = dist_reverse.sample(seed=sample_key)
            dist_forward = self.transition_distribution(dist_ind, x_hat)
            logw += dist_forward.log_prob(x) - dist_reverse.log_prob(x_hat)
            x = x_hat
            inputs = key, logw, x
            return key, logw, x

        _, logw, x = jax.lax.fori_loop(0, self._total_steps, body_fn, inputs)
        logw += self._initial_distribution.log_prob(x)
        return x, logw


def leapfrog_step(x_0: Array,
                  v_0: Array,
                  gradient_target: GradientTarget,
                  step_size: Array,
                  mass_diag_sqrt: Array,
                  num_steps: int):
    """Multiple leapfrog steps with no metropolis correction."""
    x_k = x_0
    v_k = v_0
    if mass_diag_sqrt is None:
        mass_diag_sqrt = jnp.ones_like(x_k)

    mass_diag = mass_diag_sqrt ** 2.

    for _ in range(num_steps):  # Inefficient version - should combine half steps
        v_k += 0.5 * step_size * gradient_target(x_k)  # half step in v
        x_k += step_size * v_k / mass_diag  # Step in x
        grad = gradient_target(x_k)
        v_k += 0.5 * step_size * grad  # half step in v
    return x_k, v_k


class AnnealedUHASampler:
    """Implements AIS with ULA"""

    def __init__(self,
                 num_steps: int,
                 num_samples_per_step: int,
                 step_sizes: jnp.array,
                 damping_coeff: int,
                 mass_diag_sqrt: int,
                 num_leapfrog_steps: int,
                 initial_distribution: distrax.Distribution,
                 target_distribution,
                 gradient_function):
        assert len(step_sizes) == num_steps, "Must have as many stepsizes as intermediate distributions."
        self._damping_coeff = damping_coeff
        self._mass_diag_sqrt = mass_diag_sqrt
        self._step_sizes = step_sizes
        self._num_steps = num_steps
        self._num_leapfrog_steps = num_leapfrog_steps
        self._num_samples_per_step = num_samples_per_step
        self._initial_distribution = initial_distribution
        self._target_distribution = target_distribution
        if target_distribution is None:
            assert gradient_function is not None
            self._gradient_function = gradient_function
        else:
            self._gradient_function = jax.grad(
                lambda x, i: target_distribution(x, i).sum())

        self._total_steps = self._num_samples_per_step * (self._num_steps - 1)

    def leapfrog_step(self, x, v, i):
        step_size = self._step_sizes[i]
        return leapfrog_step(x, v, lambda _x: self._gradient_function(_x, i), step_size, self._mass_diag_sqrt,
                             self._num_leapfrog_steps)

    def sample(self, key: RandomKey, n_samples: int):
        key, x_key = jax.random.split(key)
        x_k = self._initial_distribution.sample(seed=x_key, sample_shape=(n_samples,))

        v_dist = distrax.MultivariateNormalDiag(
            loc=jnp.zeros_like(x_k),
            scale_diag=jnp.ones_like(x_k) * self._mass_diag_sqrt)

        key, v_key = jax.random.split(key)
        v_k = v_dist.sample(seed=v_key)

        logw = -self._initial_distribution.log_prob(x_k)

        print(x_k.shape, v_k.shape, logw.shape)

        inputs = (key, logw, x_k, v_k)

        def body_fn(i, inputs):
            # unpack inputs
            key, logw, x_k, v_k = inputs
            dist_ind = (i // self._num_samples_per_step)

            eps_key, key = jax.random.split(key)
            eps = jax.random.normal(eps_key, x_k.shape)
            # resample momentum
            v_k_prime = v_k * self._damping_coeff + jnp.sqrt(1. - self._damping_coeff ** 2) * eps * self._mass_diag_sqrt
            # advance samples
            x_k, v_k = self.leapfrog_step(x_k, v_k_prime, dist_ind)
            # compute change in density
            logp_v_p = v_dist.log_prob(v_k_prime)
            logp_v = v_dist.log_prob(v_k)
            # update importance weights
            logw += logp_v - logp_v_p
            return key, logw, x_k, v_k

        _, logw, x_k, v_k = jax.lax.fori_loop(0, self._total_steps, body_fn, inputs)

        if self._target_distribution is not None:
            logw += self._target_distribution(x_k, self._num_steps - 1)
        else:
            logw = None

        return x_k, logw, None

    def logp_raise(self, key: RandomKey, x: jnp.array):
        logw = jnp.zeros((x.shape[0],))
        x_k = x
        v_dist = distrax.MultivariateNormalDiag(
            loc=jnp.zeros_like(x_k),
            scale_diag=jnp.ones_like(x_k) * self._mass_diag_sqrt)

        key, v_key = jax.random.split(key)
        v_k = v_dist.sample(seed=v_key)

        inputs = (key, logw, x_k, v_k)

        def body_fn(i, inputs):
            key, logw, x_k, v_k = inputs
            ind = i // self._num_samples_per_step
            dist_ind = self._num_steps - 1 - ind - 1

            eps_key, key = jax.random.split(key)
            eps = jax.random.normal(eps_key, x_k.shape)
            # resample momentum
            v_k_prime = v_k * self._damping_coeff + jnp.sqrt(1. - self._damping_coeff ** 2) * eps * self._mass_diag_sqrt
            # advance samples
            x_k, v_k = self.leapfrog_step(x_k, v_k_prime, dist_ind)

            logp_v_p = v_dist.log_prob(v_k_prime)
            logp_v = v_dist.log_prob(v_k)
            # update importance weights
            logw += logp_v - logp_v_p
            return key, logw, x_k, v_k

        _, logw, x_k, v_k = jax.lax.fori_loop(0, self._total_steps, body_fn, inputs)

        logw += self._initial_distribution.log_prob(x_k)
        return x_k, logw


class AnnealedMALASampler:
    """Implements AIS with MALA"""

    def __init__(self,
                 num_steps: int,
                 num_samples_per_step: int,
                 step_sizes: jnp.array,
                 initial_distribution: distrax.Distribution,
                 target_distribution,
                 gradient_function,
                 energy_function):
        assert len(step_sizes) == num_steps, "Must have as many stepsizes as intermediate distributions."
        self._step_sizes = step_sizes
        self._num_steps = num_steps
        self._num_samples_per_step = num_samples_per_step
        self._initial_distribution = initial_distribution
        self._target_distribution = target_distribution

        self._gradient_function = gradient_function
        self._energy_function = energy_function

        self._total_steps = self._num_samples_per_step * (self._num_steps)
        self._total_steps_reverse = self._num_samples_per_step * self._num_steps

    def transition_distribution(self, i, x):
        ss = self._step_sizes[i]
        std = (2 * ss) ** .5
        grad = self._gradient_function(x, i)
        mu = x + grad * ss
        dist = distrax.MultivariateNormalDiag(mu, jnp.ones_like(mu) * std)
        return dist

    def sample(self, key: RandomKey, n_samples: int):
        init_key, key = jax.random.split(key)
        x = self._initial_distribution.sample(seed=init_key,
                                              sample_shape=(n_samples,))
        logw = -self._initial_distribution.log_prob(x)

        accept_rate = jnp.zeros((self._num_steps,))
        inputs = (key, logw, x, accept_rate)

        def body_fn(i, inputs):
            # setup
            key, logw, x, accept_rate = inputs
            dist_ind = (i // self._num_samples_per_step)
            sample_key, accept_key, key = jax.random.split(key, 3)
            # compute forward distribution and sample
            dist_forward = self.transition_distribution(dist_ind, x)
            x_hat = dist_forward.sample(seed=sample_key)
            # compute reverse distribution
            dist_reverse = self.transition_distribution(dist_ind, x_hat)
            # compute previous and current logp(x)
            logp_x = self._energy_function(x, dist_ind)
            logp_x_hat = self._energy_function(x_hat, dist_ind)
            # compute proposal and reversal probs
            logp_reverse = dist_reverse.log_prob(x)
            logp_forward = dist_forward.log_prob(x_hat)
            # accept prob
            logp_accept = logp_x_hat - logp_x + logp_reverse - logp_forward
            u = jax.random.uniform(accept_key, (x.shape[0],))
            accept = (u < jnp.exp(logp_accept)).astype(jnp.float32)
            # update samples and importance weights
            x = accept[:, None] * x_hat + (1 - accept[:, None]) * x
            logw += (logp_x - logp_x_hat) * accept
            # update accept rate
            accept_rate = accept_rate.at[dist_ind].set(accept_rate[dist_ind] + accept.mean())
            return key, logw, x, accept_rate

        _, logw, x, accept_rate = jax.lax.fori_loop(0, self._total_steps, body_fn, inputs)
        accept_rate /= self._num_samples_per_step

        # logw += self._target_distribution(x, self._num_steps - 1)
        return x, logw, accept_rate

    def logp_raise(self, key: RandomKey, x: jnp.array):
        logw = jnp.zeros((x.shape[0],))
        accept_rate = jnp.zeros((self._num_steps,))

        inputs = (key, logw, x, accept_rate)

        def body_fn(i, inputs):
            # setup
            key, logw, x, accept_rate = inputs
            ind = i // self._num_samples_per_step
            dist_ind = self._num_steps - 1 - ind
            sample_key, accept_key, key = jax.random.split(key, 3)
            # compute reverse distribution and sample
            dist_reverse = self.transition_distribution(dist_ind, x)
            x_hat = dist_reverse.sample(seed=sample_key)
            # compute the forward distribution
            dist_forward = self.transition_distribution(dist_ind, x_hat)
            # compute previous and current logp(x)
            logp_x = self._target_distribution(x, dist_ind)
            logp_x_hat = self._target_distribution(x_hat, dist_ind)
            # compute proposal and reversal probs
            logp_reverse = dist_reverse.log_prob(x_hat)
            logp_forward = dist_forward.log_prob(x)
            # accept prob
            logp_accept = logp_x_hat - logp_x + logp_forward - logp_reverse
            u = jax.random.uniform(accept_key, (x.shape[0],))
            accept = (u < jnp.exp(logp_accept)).astype(jnp.float32)
            # update samples and importance weights
            x = accept[:, None] * x_hat + (1 - accept[:, None]) * x
            logw += (logp_x - logp_x_hat) * accept
            # update accept rate
            accept_rate = accept_rate.at[dist_ind].set(accept_rate[dist_ind] + accept.mean())
            return key, logw, x, accept_rate

        _, logw, x, accept_rate = jax.lax.fori_loop(0, self._total_steps_reverse, body_fn, inputs)
        accept_rate /= self._num_samples_per_step
        logw += self._initial_distribution.log_prob(x)
        return x, logw, accept_rate


def update_step_sizes(step_sizes, accept_rate, optimal_acc_rate=.57, lr_step_size=0.02):
    return step_sizes * (1.0 + lr_step_size * (accept_rate - optimal_acc_rate))


class AnnealedMUHASampler:
    """Implements AIS with ULA"""

    def __init__(self,
                 num_steps: int,
                 num_samples_per_step: int,
                 step_sizes: jnp.array,
                 damping_coeff: int,
                 mass_diag_sqrt: float,
                 num_leapfrog_steps: int,
                 initial_distribution: distrax.Distribution,
                 target_distribution,
                 gradient_function,
                 energy_function):
        assert len(step_sizes) == num_steps, "Must have as many stepsizes as intermediate distributions."
        self._damping_coeff = damping_coeff
        self._mass_diag_sqrt = mass_diag_sqrt
        self._step_sizes = step_sizes
        self._num_steps = num_steps
        self._num_leapfrog_steps = num_leapfrog_steps
        self._num_samples_per_step = num_samples_per_step
        self._initial_distribution = initial_distribution
        self._gradient_function = gradient_function
        self._energy_function = energy_function

        self._total_steps = self._num_samples_per_step * (self._num_steps - 1)
        self._total_steps_reverse = self._num_samples_per_step * self._num_steps

    def leapfrog_step(self, x, v, i):
        step_size = self._step_sizes[i]
        return leapfrog_step(x, v, lambda _x: self._gradient_function(_x, i), step_size, self._mass_diag_sqrt,
                             self._num_leapfrog_steps)

    def sample(self, key: RandomKey, n_samples: int):
        key, x_key = jax.random.split(key)
        x_k = self._initial_distribution.sample(seed=x_key, sample_shape=(n_samples,))

        v_dist = distrax.MultivariateNormalDiag(
            loc=jnp.zeros_like(x_k),
            scale_diag=jnp.ones_like(x_k) * self._mass_diag_sqrt)

        key, v_key = jax.random.split(key)
        v_k = v_dist.sample(seed=v_key)

        logw = -self._initial_distribution.log_prob(x_k)

        accept_rate = jnp.zeros((self._num_steps,))
        inputs = (key, logw, x_k, v_k, accept_rate)

        def body_fn(i, inputs):
            # unpack inputs
            key, logw, x_k, v_k, accept_rate = inputs
            dist_ind = (i // self._num_samples_per_step) + 1
            eps_key, accept_key, key = jax.random.split(key, 3)
            eps = jax.random.normal(eps_key, x_k.shape)
            # resample momentum
            v_k_prime = v_k * self._damping_coeff + jnp.sqrt(1. - self._damping_coeff ** 2) * eps * self._mass_diag_sqrt
            # advance samples
            x_k_next, v_k_next = self.leapfrog_step(x_k, v_k_prime, dist_ind)
            # compute change in density
            logp_v_p = v_dist.log_prob(v_k_prime)
            logp_v = v_dist.log_prob(v_k_next)
            # compute target log-probs
            logp_x = self._energy_function(x_k, dist_ind)
            logp_x_hat = self._energy_function(x_k_next, dist_ind)
            # compute joint log-probs
            log_joint_prev = logp_x + logp_v_p
            log_joint_next = logp_x_hat + logp_v
            # acceptance prob
            logp_accept = log_joint_next - log_joint_prev
            u = jax.random.uniform(accept_key, (x_k_next.shape[0],))
            accept = (u < jnp.exp(logp_accept)).astype(jnp.float32)
            # update importance weights
            logw += (logp_x - logp_x_hat) * accept
            # update samples
            x_k = accept[:, None] * x_k_next + (1 - accept[:, None]) * x_k
            v_k = accept[:, None] * v_k_next + (1 - accept[:, None]) * v_k_prime
            accept_rate = accept_rate.at[dist_ind].set(accept_rate[dist_ind] + accept.mean())
            return key, logw, x_k, v_k, accept_rate

        _, logw, x_k, v_k, accept_rate = jax.lax.fori_loop(0, self._total_steps, body_fn, inputs)

        # logw += self._target_distribution(x_k, self._num_steps - 1)
        accept_rate /= self._num_samples_per_step
        return x_k, logw, accept_rate

    def sample_gradient(self, key: RandomKey, n_samples: int):
        key, x_key = jax.random.split(key)
        x_k = self._initial_distribution.sample(seed=x_key, sample_shape=(n_samples,))

        v_dist = distrax.MultivariateNormalDiag(
            loc=jnp.zeros_like(x_k),
            scale_diag=jnp.ones_like(x_k) * self._mass_diag_sqrt)

        key, v_key = jax.random.split(key)
        v_k = v_dist.sample(seed=v_key)

        logw = -self._initial_distribution.log_prob(x_k)

        accept_rate = jnp.zeros((self._num_steps,))
        inputs = (key, logw, x_k, v_k, accept_rate)

        def body_fn(i, inputs):
            # unpack inputs
            key, logw, x_k, v_k, accept_rate = inputs
            dist_ind = (i // self._num_samples_per_step) + 1
            eps_key, accept_key, key = jax.random.split(key, 3)
            eps = jax.random.normal(eps_key, x_k.shape)
            # resample momentum
            v_k_prime = v_k * self._damping_coeff + jnp.sqrt(1. - self._damping_coeff ** 2) * eps * self._mass_diag_sqrt
            # advance samples
            x_k_next, v_k_next = self.leapfrog_step(x_k, v_k_prime, dist_ind)
            # compute change in density
            logp_v_p = v_dist.log_prob(v_k_prime)
            logp_v = v_dist.log_prob(v_k_next)
            # compute target log-probs
            logp_x = self._energy_function(x_k, dist_ind)
            logp_x_hat = self._energy_function(x_k_next, dist_ind)
            # compute joint log-probs
            log_joint_prev = logp_x + logp_v_p
            log_joint_next = logp_x_hat + logp_v
            # acceptance prob
            logp_accept = log_joint_next - log_joint_prev
            u = jax.random.uniform(accept_key, (x_k_next.shape[0],))
            accept = (u < jnp.exp(logp_accept)).astype(jnp.float32)
            # update importance weights
            logw += (logp_x - logp_x_hat) * accept
            # update samples
            x_k = accept[:, None] * x_k_next + (1 - accept[:, None]) * x_k
            v_k = accept[:, None] * v_k_next + (1 - accept[:, None]) * v_k_prime
            accept_rate = accept_rate.at[dist_ind].set(accept_rate[dist_ind] + accept.mean())
            return key, logw, x_k, v_k, accept_rate

        _, logw, x_k, v_k, accept_rate = jax.lax.fori_loop(0, self._total_steps, body_fn, inputs)

        # logw += self._target_distribution(x_k, self._num_steps - 1)
        accept_rate /= self._num_samples_per_step
        return x_k, logw, accept_rate

    def logp_raise(self, key: RandomKey, x: jnp.array):
        logw = jnp.zeros((x.shape[0],))
        x_k = x
        v_dist = distrax.MultivariateNormalDiag(
            loc=jnp.zeros_like(x_k),
            scale_diag=jnp.ones_like(x_k) * self._mass_diag_sqrt)

        key, v_key = jax.random.split(key)
        v_k = v_dist.sample(seed=v_key)

        accept_rate = jnp.zeros((self._num_steps,))

        inputs = (key, logw, x_k, v_k, accept_rate)

        def body_fn(i, inputs):
            key, logw, x_k, v_k, accept_rate = inputs
            ind = i // self._num_samples_per_step
            dist_ind = self._num_steps - 1 - ind

            eps_key, accept_key, key = jax.random.split(key, 3)
            eps = jax.random.normal(eps_key, x_k.shape)
            # resample momentum
            v_k_prime = v_k * self._damping_coeff + jnp.sqrt(1. - self._damping_coeff ** 2) * eps * self._mass_diag_sqrt
            # advance samples
            x_k_next, v_k_next = self.leapfrog_step(x_k, v_k_prime, dist_ind)
            # compute change in density
            logp_v_p = v_dist.log_prob(v_k_prime)
            logp_v = v_dist.log_prob(v_k_next)
            # compute target log-probs
            logp_x = self._target_distribution(x_k, dist_ind)
            logp_x_hat = self._target_distribution(x_k_next, dist_ind)
            # compute joint log-probs
            log_joint_prev = logp_x + logp_v_p
            log_joint_next = logp_x_hat + logp_v
            # acceptance prob
            logp_accept = log_joint_next - log_joint_prev
            u = jax.random.uniform(accept_key, (x.shape[0],))
            accept = (u < jnp.exp(logp_accept)).astype(jnp.float32)
            logw += (logp_x - logp_x_hat) * accept
            # update samples
            x_k = accept[:, None] * x_k_next + (1 - accept[:, None]) * x_k
            v_k = accept[:, None] * v_k_next + (1 - accept[:, None]) * v_k_prime
            accept_rate = accept_rate.at[dist_ind].set(accept_rate[dist_ind] + accept.mean())
            return key, logw, x_k, v_k, accept_rate

        _, logw, x_k, v_k, accept_rate = jax.lax.fori_loop(0, self._total_steps_reverse, body_fn, inputs)

        logw += self._initial_distribution.log_prob(x_k)
        accept_rate /= self._num_samples_per_step
        return x_k, logw, accept_rate


class AnnealedMUHADiffSampler:
    """Implements AIS with ULA"""

    def __init__(self,
                 num_steps: int,
                 num_samples_per_step: int,
                 step_sizes: jnp.array,
                 damping_coeff: int,
                 mass_diag_sqrt: float,
                 num_leapfrog_steps: int,
                 initial_distribution: distrax.Distribution,
                 target_distribution,
                 gradient_function,
                 energy_function,
                 n_trapets: int = 5):
        assert len(step_sizes) == num_steps, "Must have as many stepsizes as intermediate distributions."
        self._damping_coeff = damping_coeff
        self._mass_diag_sqrt = mass_diag_sqrt
        self._step_sizes = step_sizes
        self._num_steps = num_steps
        self._num_leapfrog_steps = num_leapfrog_steps
        self._num_samples_per_step = num_samples_per_step
        self._initial_distribution = initial_distribution
        self._gradient_function = gradient_function
        self._energy_function = energy_function

        self._total_steps = self._num_samples_per_step * (self._num_steps - 1)
        self._total_steps_reverse = self._num_samples_per_step * self._num_steps
        self.n_trapets = n_trapets

    def leapfrog_step(self, x, v, i):
        step_size = self._step_sizes[i]
        return leapfrog_step(x, v, lambda _x: self._gradient_function(_x, i), step_size, self._mass_diag_sqrt,
                             self._num_leapfrog_steps)

    def sample(self, key: RandomKey, n_samples: int):
        key, x_key = jax.random.split(key)
        x_k = self._initial_distribution.sample(seed=x_key, sample_shape=(n_samples,))

        v_dist = distrax.MultivariateNormalDiag(
            loc=jnp.zeros_like(x_k),
            scale_diag=jnp.ones_like(x_k) * self._mass_diag_sqrt)

        key, v_key = jax.random.split(key)
        v_k = v_dist.sample(seed=v_key)

        logw = -self._initial_distribution.log_prob(x_k)

        accept_rate = jnp.zeros((self._num_steps,))
        inputs = (key, logw, x_k, v_k, accept_rate)

        def body_fn(i, inputs):
            # unpack inputs
            key, logw, x_k, v_k, accept_rate = inputs
            dist_ind = (i // self._num_samples_per_step) + 1
            eps_key, accept_key, key = jax.random.split(key, 3)
            eps = jax.random.normal(eps_key, x_k.shape)
            # resample momentum
            v_k_prime = v_k * self._damping_coeff + jnp.sqrt(1. - self._damping_coeff ** 2) * eps * self._mass_diag_sqrt
            # advance samples
            x_k_next, v_k_next = self.leapfrog_step(x_k, v_k_prime, dist_ind)
            # compute change in density
            logp_v_p = v_dist.log_prob(v_k_prime)
            logp_v = v_dist.log_prob(v_k_next)
            # compute target log-probs

            if self.n_trapets > 0:
                def energy_s(ss):
                    diff = x_k_next - x_k
                    x = x_k + ss[0] * diff
                    energy = jnp.sum(self._energy_function(x=x, t=dist_ind) * diff, axis=1).reshape(-1, 1)
                    print(energy)
                    for i in range(1, len(ss)):
                        x = x_k + ss[i] * diff
                        energy = jnp.concatenate(
                            (energy, jnp.sum(self._energy_function(x=x, t=dist_ind) * diff, axis=1).reshape(-1, 1)),
                            axis=1)
                    return energy

                s = jnp.linspace(0, 1, num=self.n_trapets)
                diff_logp_x = jnp.trapz(energy_s(s), s)
            else:
                # Taylor-Expansion
                ef = partial(self._energy_function, t=dist_ind)
                diff = x_k_next-x_k
                _, jvp_x = jvp(ef, (x_k,), (diff,))
                diff_logp_x = jnp.sum((0.5*jvp_x + ef(x_k)) * diff, axis=1)

            # compute joint log-probs
            # acceptance prob
            logp_accept = logp_v - logp_v_p + diff_logp_x
            u = jax.random.uniform(accept_key, (x_k_next.shape[0],))
            accept = (u < jnp.exp(logp_accept)).astype(jnp.float32)
            # update importance weights
            logw += diff_logp_x * accept
            # update samples
            x_k = accept[:, None] * x_k_next + (1 - accept[:, None]) * x_k
            v_k = accept[:, None] * v_k_next + (1 - accept[:, None]) * v_k_prime
            accept_rate = accept_rate.at[dist_ind].set(accept_rate[dist_ind] + accept.mean())
            return key, logw, x_k, v_k, accept_rate

        _, logw, x_k, v_k, accept_rate = jax.lax.fori_loop(0, self._total_steps, body_fn, inputs)

        # logw += self._target_distribution(x_k, self._num_steps - 1)
        accept_rate /= self._num_samples_per_step
        return x_k, logw, accept_rate

    def sample_gradient(self, key: RandomKey, n_samples: int):
        key, x_key = jax.random.split(key)
        x_k = self._initial_distribution.sample(seed=x_key, sample_shape=(n_samples,))

        v_dist = distrax.MultivariateNormalDiag(
            loc=jnp.zeros_like(x_k),
            scale_diag=jnp.ones_like(x_k) * self._mass_diag_sqrt)

        key, v_key = jax.random.split(key)
        v_k = v_dist.sample(seed=v_key)

        logw = -self._initial_distribution.log_prob(x_k)

        accept_rate = jnp.zeros((self._num_steps,))
        inputs = (key, logw, x_k, v_k, accept_rate)

        def body_fn(i, inputs):
            # unpack inputs
            key, logw, x_k, v_k, accept_rate = inputs
            dist_ind = (i // self._num_samples_per_step) + 1
            eps_key, accept_key, key = jax.random.split(key, 3)
            eps = jax.random.normal(eps_key, x_k.shape)
            # resample momentum
            v_k_prime = v_k * self._damping_coeff + jnp.sqrt(1. - self._damping_coeff ** 2) * eps * self._mass_diag_sqrt
            # advance samples
            x_k_next, v_k_next = self.leapfrog_step(x_k, v_k_prime, dist_ind)
            # compute change in density
            logp_v_p = v_dist.log_prob(v_k_prime)
            logp_v = v_dist.log_prob(v_k_next)
            # compute target log-probs
            logp_x = self._energy_function(x_k, dist_ind)
            logp_x_hat = self._energy_function(x_k_next, dist_ind)
            # compute joint log-probs
            log_joint_prev = logp_x + logp_v_p
            log_joint_next = logp_x_hat + logp_v
            # acceptance prob
            logp_accept = log_joint_next - log_joint_prev
            u = jax.random.uniform(accept_key, (x_k_next.shape[0],))
            accept = (u < jnp.exp(logp_accept)).astype(jnp.float32)
            # update importance weights
            logw += (logp_x - logp_x_hat) * accept
            # update samples
            x_k = accept[:, None] * x_k_next + (1 - accept[:, None]) * x_k
            v_k = accept[:, None] * v_k_next + (1 - accept[:, None]) * v_k_prime
            accept_rate = accept_rate.at[dist_ind].set(accept_rate[dist_ind] + accept.mean())
            return key, logw, x_k, v_k, accept_rate

        _, logw, x_k, v_k, accept_rate = jax.lax.fori_loop(0, self._total_steps, body_fn, inputs)

        # logw += self._target_distribution(x_k, self._num_steps - 1)
        accept_rate /= self._num_samples_per_step
        return x_k, logw, accept_rate

    def logp_raise(self, key: RandomKey, x: jnp.array):
        logw = jnp.zeros((x.shape[0],))
        x_k = x
        v_dist = distrax.MultivariateNormalDiag(
            loc=jnp.zeros_like(x_k),
            scale_diag=jnp.ones_like(x_k) * self._mass_diag_sqrt)

        key, v_key = jax.random.split(key)
        v_k = v_dist.sample(seed=v_key)

        accept_rate = jnp.zeros((self._num_steps,))

        inputs = (key, logw, x_k, v_k, accept_rate)

        def body_fn(i, inputs):
            key, logw, x_k, v_k, accept_rate = inputs
            ind = i // self._num_samples_per_step
            dist_ind = self._num_steps - 1 - ind

            eps_key, accept_key, key = jax.random.split(key, 3)
            eps = jax.random.normal(eps_key, x_k.shape)
            # resample momentum
            v_k_prime = v_k * self._damping_coeff + jnp.sqrt(1. - self._damping_coeff ** 2) * eps * self._mass_diag_sqrt
            # advance samples
            x_k_next, v_k_next = self.leapfrog_step(x_k, v_k_prime, dist_ind)
            # compute change in density
            logp_v_p = v_dist.log_prob(v_k_prime)
            logp_v = v_dist.log_prob(v_k_next)
            # compute target log-probs
            logp_x = self._target_distribution(x_k, dist_ind)
            logp_x_hat = self._target_distribution(x_k_next, dist_ind)
            # compute joint log-probs
            log_joint_prev = logp_x + logp_v_p
            log_joint_next = logp_x_hat + logp_v
            # acceptance prob
            logp_accept = log_joint_next - log_joint_prev
            u = jax.random.uniform(accept_key, (x.shape[0],))
            accept = (u < jnp.exp(logp_accept)).astype(jnp.float32)
            logw += (logp_x - logp_x_hat) * accept
            # update samples
            x_k = accept[:, None] * x_k_next + (1 - accept[:, None]) * x_k
            v_k = accept[:, None] * v_k_next + (1 - accept[:, None]) * v_k_prime
            accept_rate = accept_rate.at[dist_ind].set(accept_rate[dist_ind] + accept.mean())
            return key, logw, x_k, v_k, accept_rate

        _, logw, x_k, v_k, accept_rate = jax.lax.fori_loop(0, self._total_steps_reverse, body_fn, inputs)

        logw += self._initial_distribution.log_prob(x_k)
        accept_rate /= self._num_samples_per_step
        return x_k, logw, accept_rate
