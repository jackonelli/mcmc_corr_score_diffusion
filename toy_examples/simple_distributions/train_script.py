import jax
import jax.numpy as jnp
import haiku as hk
import distrax
import chex
import numpy as np
import optax
import matplotlib.pyplot as plt
import time
import pickle
import tensorflow as tf
import tensorflow_datasets as tfds
import functools
import pandas as pd
from reduce_reuse_recycle.toy_examples.simple_distributions.datasets import toy_gmm
from reduce_reuse_recycle.toy_examples.simple_distributions.models import ResnetDiffusionModel, EBMDiffusionModel, \
    PortableDiffusionModel


# Train Spiral EBM Model
batch_size = 1000
data_dim = 2
num_steps = 15001
ebm = True

EMA = .999

n_steps = 100
net_params = {"n_layers": 4,
              "h_dim": 128,
              "emb_dim": 32}

xr = [-.75, .75]
yr = [-.75, .75]


def forward_fn():
    net = ResnetDiffusionModel(n_steps=n_steps, n_layers=4, x_dim=data_dim, h_dim=128, emb_dim=32)

    if ebm:
        net = EBMDiffusionModel(net)

    ddpm = PortableDiffusionModel(data_dim, n_steps, net, var_type="beta_forward")

    def logp_unnorm(x, t):
        scale_e = ddpm.energy_scale(-2 - t)
        t = jnp.ones((x.shape[0],), dtype=jnp.int32) * t
        return -net.neg_logp_unnorm(x, t) * scale_e

    def _logpx(x):
        return ddpm.logpx(x)["logpx"]

    return ddpm.loss, (ddpm.loss, ddpm.sample, _logpx, logp_unnorm)


def plot_samples(x):
    plt.scatter(x[:, 0], x[:, 1])
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)


def dist_show_2d(fn, xr, yr):
    nticks = 100
    x, y = np.meshgrid(np.linspace(xr[0], xr[1], nticks), np.linspace(yr[0], yr[1], nticks))
    coord = np.stack([x, y], axis=-1).reshape((-1, 2))
    heatmap = fn(coord).reshape((nticks, nticks))
    plt.imshow(heatmap)


if __name__ == '__main__':
    # region
    load = True
    forward = hk.multi_transform(forward_fn)
    rng_seq = hk.PRNGSequence(0)

    # load data
    dataset_energy, dataset_sample = toy_gmm(std=.03)
    x = dataset_sample(batch_size)

    plot_samples(x)
    plt.show()
    x = x.reshape(x.shape[0], -1)

    params = forward.init(next(rng_seq), x)
    loss_fn, sample_fn, logpx_fn, logp_unnorm_fn = forward.apply
    param_count = sum(x.size for x in jax.tree_leaves(params))
    for k, v in jax.tree_map(lambda x: x.shape, params).items():
        print(k, v)
    print("Model has {} params".format(param_count))

    opt = optax.adam(1e-3)
    opt_state = opt.init(params)

    sample_fn = jax.jit(sample_fn, static_argnums=2)
    logpx_fn = jax.jit(logpx_fn)

    logp_unnorm_fn = jax.jit(logp_unnorm_fn)


    @jax.jit
    def mean_loss_fn(params, rng, x):
        loss = loss_fn(params, rng, x)
        return loss.mean()


    @jax.jit
    def update(params, opt_state, rng, x):
        loss, grad = jax.value_and_grad(mean_loss_fn)(params, rng, x)

        updates, opt_state = opt.update(grad, opt_state)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, opt_state


    ema_params = params
    losses = []
    test_logpx = []
    itr = 0

    if load:
        params = pickle.load(open("params_ebmdiff.p", "rb"))
        ema_params = params
        x = dataset_sample(batch_size)

        x = x.reshape(x.shape[0], -1)

        x_samp = sample_fn(ema_params, next(rng_seq), batch_size)
        plot_samples(x_samp)
        plt.show()
        if ebm:
            dist_show_2d(lambda x: logp_unnorm_fn(ema_params, next(rng_seq), x, 0), xr=xr, yr=yr)
            plt.title(str(itr))
            plt.show()
    else:
        for itr in range(num_steps):
            x = dataset_sample(batch_size)

            x = x.reshape(x.shape[0], -1)
            start_time = time.time()
            loss, params, opt_state = update(params, opt_state, next(rng_seq), x)
            duration_update = time.time() - start_time
            ema_params = jax.tree_map(lambda e, p: e * EMA + p * (1 - EMA), ema_params, params)

            if itr % 100 == 0:
                print(itr, loss, "time:", duration_update)
                losses.append(loss)
            if itr % 1000 == 0:
                x_samp = sample_fn(ema_params, next(rng_seq), batch_size)
                plot_samples(x_samp)
                plt.show()
                logpx = logpx_fn(ema_params, next(rng_seq), x).mean()
                print("TEST", itr, 'logpx', logpx)
                test_logpx.append(logpx)

                if ebm:
                    dist_show_2d(lambda x: logp_unnorm_fn(ema_params, next(rng_seq), x, 0), xr=xr, yr=yr)
                    plt.title(str(itr))
                    plt.show()
                    """
                    for t in range(10):
                        dist_show_2d(lambda x: logp_unnorm_fn(ema_params, next(rng_seq), x, 10 * t), xr=xr, yr=yr)
                        plt.show()
                    """
        plt.plot(losses)
        plt.show()
        plt.plot(test_logpx)
    # endregion

    if ebm:
        _, dual_product_sample_fn, dual_product_nll, dual_product_logp_unorm_fn, dual_product_gradient_fn, dual_product_energy_fn = forward_product.apply
    else:
        _, dual_product_sample_fn, dual_product_nll, dual_product_logp_unorm_fn, dual_product_gradient_fn = forward_product.apply

    dual_product_sample_fn = jax.jit(dual_product_sample_fn, static_argnums=2)
    dual_product_logp_unnorm_fn = jax.jit(dual_product_logp_unorm_fn)
    dual_product_gradient_fn = jax.jit(dual_product_gradient_fn)

    if ebm:
        dual_product_energy_fn = jax.jit(dual_product_energy_fn)

    dual_product_nll = jax.jit(dual_product_nll)

    # Sampling from product of distributions
    dim = 2
    n_mode = 4
    std = .05
    init_std = 1.
    init_mu = 0.
    n_steps = 10
    damping = .5
    mass_diag_sqrt = 1.
    num_leapfrog = 3
    samples_per_step = 10
    uha_step_size = .03
    ula_step_size = .001

    batch_size = 1000

    rng_seq = hk.PRNGSequence(0)

    means = jax.random.normal(next(rng_seq), (n_mode, dim))
    comp_dists = distrax.MultivariateNormalDiag(means, jnp.ones_like(means) * std)
    pi = distrax.Categorical(logits=jnp.zeros((n_mode,)))
    target_dist = distrax.MixtureSameFamily(pi, comp_dists)
    initial_dist = distrax.MultivariateNormalDiag(means[0] * 0 + init_mu, init_std * jnp.ones_like(means[0]))

    x_init = initial_dist.sample(seed=next(rng_seq), sample_shape=(batch_size,))
    logZs = []
    n_stepss = [100]


    def gradient_function(x, t):
        t = n_steps - jnp.ones((x.shape[0],), dtype=jnp.int32) * t - 1
        return -1 * dual_product_gradient_fn(params, next(rng_seq), x, t)


    if ebm:
        def energy_function(x, t):
            t = n_steps - jnp.ones((x.shape[0],), dtype=jnp.int32) * t - 1
            return -1 * dual_product_energy_fn(params, next(rng_seq), x, t)

    for n_steps in n_stepss:
        ula_step_sizes = jnp.ones((n_steps,)) * ula_step_size
        uha_step_sizes = jnp.ones((n_steps,)) * uha_step_size

        betas = jnp.linspace(0., 1., n_steps)


        def target_function(x, i):
            beta = betas[i]
            init_lp = initial_dist.log_prob(x)
            targ_lp = target_dist.log_prob(x)
            return beta * targ_lp + (1 - beta) * init_lp


        # Choose MCMC Sampler to use
        # sampler = AnnealedULASampler(n_steps, samples_per_step, ula_step_sizes, initial_dist, target_distribution=None, gradient_function=gradient_function)
        # sampler = AnnealedMALASampler(n_steps, samples_per_step, ula_step_sizes, initial_dist, target_distribution=None, gradient_function=gradient_function, energy_function=energy_function)
        # sampler = AnnealedUHASampler(n_steps, samples_per_step, uha_step_sizes, damping, mass_diag_sqrt, num_leapfrog, initial_dist, target_distribution=None, gradient_function=gradient_function)
        sampler = AnnealedMUHASampler(n_steps, samples_per_step, uha_step_sizes, damping, mass_diag_sqrt, num_leapfrog,
                                      initial_dist, target_distribution=None, gradient_function=gradient_function,
                                      energy_function=energy_function)

        x_samp, logw, accept = sampler.sample(next(rng_seq), batch_size)
        rng_seq = hk.PRNGSequence(1)
        grad_sample = dual_product_sample_fn(params, next(rng_seq), batch_size, jnp.inf)

        # Samples from MCMC
        plt.scatter(x_samp[:, 0], x_samp[:, 1], color='green', alpha=.5)

        # Samples from adding score functions
        plt.scatter(grad_sample[:, 0], grad_sample[:, 1], color='blue', alpha=.5)

        plt.show()