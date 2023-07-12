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


# Define different datasets to train diffusion models
def toy_gmm(n_comp=8, std=0.075, radius=0.5):
    """Ring of 2D Gaussians. Returns energy and sample functions."""

    means_x = np.cos(2 * np.pi *
                     np.linspace(0, (n_comp - 1) / n_comp, n_comp)).reshape(
        n_comp, 1, 1, 1)
    means_y = np.sin(2 * np.pi *
                     np.linspace(0, (n_comp - 1) / n_comp, n_comp)).reshape(
        n_comp, 1, 1, 1)
    mean = radius * np.concatenate((means_x, means_y), axis=1)
    weights = np.ones(n_comp) / n_comp

    def nll(x):
        means = jnp.array(mean.reshape((-1, 1, 2)))
        c = np.log(n_comp * 2 * np.pi * std ** 2)
        f = jax.nn.logsumexp(
            jnp.sum(-0.5 * jnp.square((x - means) / std), axis=2), axis=0) + c
        # f = f + np.log(2)

        return -f

    def sample(n_samples):
        toy_sample = np.zeros(0).reshape((0, 2, 1, 1))
        sample_group_sz = np.random.multinomial(n_samples, weights)
        for i in range(n_comp):
            sample_group = mean[i] + std * np.random.randn(
                2 * sample_group_sz[i]).reshape(-1, 2, 1, 1)
            toy_sample = np.concatenate((toy_sample, sample_group), axis=0)
            np.random.shuffle(toy_sample)
        data = toy_sample[:, :, 0, 0]

        return data

    def _sample_constraint(n_samples, x_interval, y_interval):
        samples = sample(n_samples)
        x_accept = np.logical_and(samples[:, 0] > x_interval[0], samples[:, 0] < x_interval[1])
        y_accept = np.logical_and(samples[:, 1] > y_interval[0], samples[:, 1] < y_interval[1])
        return samples[np.logical_and(x_accept, y_accept)]

    def sample_constraint(n_samples, x_interval=None, y_interval=None):
        if x_interval is None:
            x_interval = [-np.inf, np.inf]

        if y_interval is None:
            y_interval = [-np.inf, np.inf]

        samples = _sample_constraint(n_samples, x_interval, y_interval)
        n = samples.shape[0]
        while n < n_samples:
            samples_ = _sample_constraint(n_samples, x_interval, y_interval)
            samples = np.concatenate((samples, samples_))
            n = samples.shape[0]
        return samples[:n_samples]

    return nll, sample_constraint, mean


def toy_gauss(radius=0.5):
    """Ring of 2D Gaussians. Returns energy and sample functions."""

    std = radius

    def nll(x):
        c = np.log(2 * np.pi * std ** 2)
        f = -0.5 * jnp.square((x) / std) + c

        return f

    def sample(n_samples):
        data = np.random.randn(n_samples, 2) * radius
        return data

    return nll, sample


def toy_box(scale=1.0):
    """Ring of 2D Gaussians. Returns energy and sample functions."""

    def nll(x):
        return 1

    def sample(n_samples):
        data = np.random.uniform(-scale, scale, (n_samples, 2))
        return data

    return nll, sample


def bar(scale=0.2, prob_inside=0.99, r=1.1):
    """Ring of 2D Gaussians. Returns energy and sample functions."""
    p_re = 1. - prob_inside
    r = 1.1
    pdf_outer = p_re / (4*r ** 2 - 4 * scale)
    pdf_inner = prob_inside/(4 * scale)

    def nll(x):
        l_x = np.zeros(shape=x.shape[0])
        l_x[(x[:, 0] > -r) & (x[:, 0] < r) & (x[:, 1] > -r) & (x[:, 1] < r)] = pdf_outer
        l_x[(x[:, 0] > -scale) & (x[:, 0] < scale) & (x[:, 1] > -1.) & (x[:, 1] < 1.)] = pdf_inner
        return -np.log(l_x)

    def sample(n_samples):
        data = np.random.uniform(-1, 1, (n_samples, 2))
        data[:, 0] = data[:, 0] * scale
        return data

    return nll, sample, pdf_outer, pdf_inner


def bar_horizontal(scale=0.2):
    """Ring of 2D Gaussians. Returns energy and sample functions."""

    def nll(x):
        return 1

    def sample(n_samples):
        data = np.random.uniform(-1, 1, (n_samples, 2))
        data[:, 1] = data[:, 1] * scale

        return data

    return nll, sample


def toy_gmm_left(n_comp=8, std=0.075, radius=0.5):
    """Ring of 2D Gaussians. Returns energy and sample functions."""

    means_x = np.cos(2 * np.pi *
                     np.linspace(0, (n_comp - 1) / n_comp, n_comp)).reshape(
        n_comp, 1, 1, 1)
    means_y = np.sin(2 * np.pi *
                     np.linspace(0, (n_comp - 1) / n_comp, n_comp)).reshape(
        n_comp, 1, 1, 1)
    mean = radius * np.concatenate((means_x, means_y), axis=1)
    mean = mean[[0, 1, 2, 3]]

    n_comp = mean.shape[0]

    weights = np.ones(n_comp) / n_comp

    def nll(x):
        means = jnp.array(mean.reshape((-1, 1, 2)))
        c = np.log(n_comp * 2 * np.pi * std ** 2)
        f = jax.nn.logsumexp(
            jnp.sum(-0.5 * jnp.square((x - means) / std), axis=2), axis=0) + c
        # f = f + np.log(2)

        return f

    def sample(n_samples):
        toy_sample = np.zeros(0).reshape((0, 2, 1, 1))
        sample_group_sz = np.random.multinomial(n_samples, weights)
        for i in range(n_comp):
            sample_group = mean[i] + std * np.random.randn(
                2 * sample_group_sz[i]).reshape(-1, 2, 1, 1)
            toy_sample = np.concatenate((toy_sample, sample_group), axis=0)
            np.random.shuffle(toy_sample)
        data = toy_sample[:, :, 0, 0]

        return data

    return nll, sample


def toy_gmm_right(n_comp=8, std=0.075, radius=0.5):
    """Ring of 2D Gaussians. Returns energy and sample functions."""

    means_x = np.cos(2 * np.pi *
                     np.linspace(0, (n_comp - 1) / n_comp, n_comp)).reshape(
        n_comp, 1, 1, 1)
    means_y = np.sin(2 * np.pi *
                     np.linspace(0, (n_comp - 1) / n_comp, n_comp)).reshape(
        n_comp, 1, 1, 1)
    mean = radius * np.concatenate((means_x, means_y), axis=1)
    mean = mean[[4, 5, 6, 7]]
    n_comp = mean.shape[0]

    weights = np.ones(n_comp) / n_comp

    def nll(x):
        means = jnp.array(mean.reshape((-1, 1, 2)))
        c = np.log(n_comp * 2 * np.pi * std ** 2)
        f = jax.nn.logsumexp(
            jnp.sum(-0.5 * jnp.square((x - means) / std), axis=2), axis=0) + c
        # f = f + np.log(2)

        return f

    def sample(n_samples):
        toy_sample = np.zeros(0).reshape((0, 2, 1, 1))
        sample_group_sz = np.random.multinomial(n_samples, weights)
        for i in range(n_comp):
            sample_group = mean[i] + std * np.random.randn(
                2 * sample_group_sz[i]).reshape(-1, 2, 1, 1)
            toy_sample = np.concatenate((toy_sample, sample_group), axis=0)
            np.random.shuffle(toy_sample)
        data = toy_sample[:, :, 0, 0]

        return data

    return nll, sample


def right_bar(scale=0.1):
    """Ring of 2D Gaussians. Returns energy and sample functions."""

    def nll(x):
        return 1

    def sample(n_samples):
        data = np.random.uniform(-1, 1, (n_samples, 2))
        data[:, 0] = data[:, 0] * scale + 0.2

        return data

    return nll, sample


def left_bar(scale=0.1):
    """Ring of 2D Gaussians. Returns energy and sample functions."""

    def nll(x):
        return 1

    def sample(n_samples):
        data = np.random.uniform(-1, 1, (n_samples, 2))
        data[:, 0] = data[:, 0] * scale - 0.2

        return data

    return nll, sample
