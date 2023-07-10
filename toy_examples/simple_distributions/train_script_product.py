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
from functools import partial
from reduce_reuse_recycle.toy_examples.simple_distributions.datasets import toy_gmm, bar
from reduce_reuse_recycle.toy_examples.simple_distributions.models import ResnetDiffusionModel, EBMDiffusionModel, \
    PortableDiffusionModel, ProductEBMDiffusionModel
from reduce_reuse_recycle.toy_examples.simple_distributions.sampler import AnnealedMUHASampler, AnnealedMUHADiffSampler, AnnealedULASampler, AnnealedUHASampler


# Train Spiral EBM Model
batch_size = 1000
data_dim = 2
num_steps = 15001

EMA = .999

n_steps = 100
net_params = {"n_layers": 4,
              "h_dim": 128,
              "emb_dim": 32}

xr = [-.75, .75]
yr = [-.75, .75]


def forward_fn(ebm=True):
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


def forward_fn_product(ebm=True):
    net_one = ResnetDiffusionModel(n_steps=n_steps, n_layers=4, x_dim=data_dim, h_dim=128, emb_dim=32)

    if ebm:
        net_one = EBMDiffusionModel(net_one)

    net_two = ResnetDiffusionModel(n_steps=n_steps, n_layers=4, x_dim=data_dim, h_dim=128, emb_dim=32)

    if ebm:
        net_two = EBMDiffusionModel(net_two)

    dual_net = ProductEBMDiffusionModel(net_one, net_two)
    ddpm = PortableDiffusionModel(data_dim, n_steps, dual_net, var_type="beta_forward")

    def logp_unnorm(x, t):
        scale_e = ddpm.energy_scale(-2 - t)
        t = jnp.ones((x.shape[0],), dtype=jnp.int32) * t
        return -dual_net.neg_logp_unnorm(x, t) * scale_e

    def _logpx(x):
        return ddpm.logpx(x)["logpx"]

    if ebm:
        return ddpm.loss, (ddpm.loss, ddpm.sample, _logpx, logp_unnorm, ddpm.p_gradient, ddpm.p_energy)
    else:
        return ddpm.loss, (ddpm.loss, ddpm.sample, _logpx, logp_unnorm, ddpm.p_gradient, ddpm.p_gradient)


def forward_fn_product_multiple():
    # EBM
    net_one_ebm = ResnetDiffusionModel(n_steps=n_steps, n_layers=4, x_dim=data_dim, h_dim=128, emb_dim=32)
    net_one_ebm = EBMDiffusionModel(net_one_ebm)

    net_two_ebm = ResnetDiffusionModel(n_steps=n_steps, n_layers=4, x_dim=data_dim, h_dim=128, emb_dim=32)
    net_two_ebm = EBMDiffusionModel(net_two_ebm)

    dual_net_ebm = ProductEBMDiffusionModel(net_one_ebm, net_two_ebm)
    ddpm_ebm = PortableDiffusionModel(data_dim, n_steps, dual_net_ebm, var_type="beta_forward")

    def logp_unnorm_ebm(x, t):
        scale_e = ddpm_ebm.energy_scale(-2 - t)
        t = jnp.ones((x.shape[0],), dtype=jnp.int32) * t
        return -dual_net_ebm.neg_logp_unnorm(x, t) * scale_e

    def _logpx_ebm(x):
        return ddpm_ebm.logpx(x)["logpx"]

    # Diffusion
    net_one_diff = ResnetDiffusionModel(n_steps=n_steps, n_layers=4, x_dim=data_dim, h_dim=128, emb_dim=32)
    net_one_diff = EBMDiffusionModel(net_one_diff)

    net_two_diff = ResnetDiffusionModel(n_steps=n_steps, n_layers=4, x_dim=data_dim, h_dim=128, emb_dim=32)
    net_two_diff = EBMDiffusionModel(net_two_diff)

    dual_net_diff = ProductEBMDiffusionModel(net_one_diff, net_two_diff)
    ddpm_diff = PortableDiffusionModel(data_dim, n_steps, dual_net_diff, var_type="beta_forward")

    def logp_unnorm_diff(x, t):
        scale_e = ddpm_diff.energy_scale(-2 - t)
        t = jnp.ones((x.shape[0],), dtype=jnp.int32) * t
        return -dual_net_diff.neg_logp_unnorm(x, t) * scale_e

    def _logpx_diff(x):
        return ddpm_diff.logpx(x)["logpx"]

    return None, (ddpm_ebm.loss, ddpm_ebm.sample, _logpx_ebm, logp_unnorm_ebm, ddpm_ebm.p_gradient, ddpm_ebm.p_energy,
                  ddpm_diff.loss, ddpm_diff.sample, _logpx_diff, logp_unnorm_diff, ddpm_diff.p_gradient, ddpm_diff.forward)


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


def train_single_model(dataset_sample, load_param=None, save_param=None, ebm=True):
    if load_param is not None:
        params = pickle.load(open(load_param, "rb"))
        return params

    partial_forward_fn = partial(forward_fn, ebm)

    forward = hk.multi_transform(partial_forward_fn)
    rng_seq = hk.PRNGSequence(0)

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
    if save_param is not None:
        pickle.dump(ema_params, open(save_param, "wb"))
    return ema_params


def sampling_product_distribution(params, ebm=True, n_trapets=5, grad=False):
    partial_forward_fn_product = partial(forward_fn_product, ebm)
    forward_product = hk.multi_transform(partial_forward_fn_product)
    rng_seq = hk.PRNGSequence(0)

    if ebm:
        _, dual_product_sample_fn, dual_product_nll, dual_product_logp_unorm_fn, dual_product_gradient_fn, dual_product_energy_fn = forward_product.apply
    else:
        _, dual_product_sample_fn, dual_product_nll, dual_product_logp_unorm_fn, dual_product_gradient_fn, dual_product_energy_fn = forward_product.apply

    dual_product_sample_fn = jax.jit(dual_product_sample_fn, static_argnums=2)
    dual_product_gradient_fn = jax.jit(dual_product_gradient_fn)

    dual_product_energy_fn = jax.jit(dual_product_energy_fn)

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

    means = jax.random.normal(next(rng_seq), (n_mode, dim))
    comp_dists = distrax.MultivariateNormalDiag(means, jnp.ones_like(means) * std)
    pi = distrax.Categorical(logits=jnp.zeros((n_mode,)))
    target_dist = distrax.MixtureSameFamily(pi, comp_dists)
    initial_dist = distrax.MultivariateNormalDiag(means[0] * 0 + init_mu, init_std * jnp.ones_like(means[0]))

    n_stepss = [100]

    def gradient_function(x, t):
        t = n_steps - jnp.ones((x.shape[0],), dtype=jnp.int32) * t - 1
        return -1 * dual_product_gradient_fn(params, next(rng_seq), x, t)

    def energy_function(x, t):
        t = n_steps - jnp.ones((x.shape[0],), dtype=jnp.int32) * t - 1
        return -1 * dual_product_energy_fn(params, next(rng_seq), x, t)

    for n_steps in n_stepss:
        ula_step_sizes = jnp.ones((n_steps,)) * ula_step_size
        uha_step_sizes = jnp.ones((n_steps,)) * uha_step_size

        betas = jnp.linspace(0., 1., n_steps)

        if ebm:
            # Choose MCMC Sampler to use
            # sampler = AnnealedULASampler(n_steps, samples_per_step, ula_step_sizes, initial_dist, target_distribution=None, gradient_function=gradient_function)
            # sampler = AnnealedMALASampler(n_steps, samples_per_step, ula_step_sizes, initial_dist, target_distribution=None, gradient_function=gradient_function, energy_function=energy_function)
            sampler = AnnealedUHASampler(n_steps, samples_per_step, uha_step_sizes, damping, mass_diag_sqrt, num_leapfrog, initial_dist, target_distribution=None, gradient_function=gradient_function)
            # sampler = AnnealedMUHASampler(n_steps, samples_per_step, uha_step_sizes, damping, mass_diag_sqrt, num_leapfrog, initial_dist, target_distribution=energy_function, gradient_function=gradient_function,energy_function=energy_function)
        else:
            sampler = AnnealedMUHADiffSampler(n_steps, samples_per_step, uha_step_sizes, damping, mass_diag_sqrt,
                                              num_leapfrog, initial_dist, target_distribution=None,
                                              gradient_function=gradient_function,
                                              energy_function=energy_function, n_trapets=n_trapets)

        x_samp, logw, accept = sampler.sample(next(rng_seq), batch_size)
        # Samples from MCMC
        plt.scatter(x_samp[:, 0], x_samp[:, 1], color='green', alpha=.5)

        rng_seq = hk.PRNGSequence(1)
        grad_sample = None
        if grad:
            grad_sample = dual_product_sample_fn(params, next(rng_seq), batch_size, jnp.inf)

            # Samples from adding score functions
            plt.scatter(grad_sample[:, 0], grad_sample[:, 1], color='blue', alpha=.5)

        plt.show()

        return x_samp, grad_sample, logw


if __name__ == '__main__':
    with jax.disable_jit():
        # Load Data and Train Model - Energy-Based Diffusion Model and Diffusion Model
        nll_gmm, dataset_sample = toy_gmm(std=.03)
        spiral_params_ebm = train_single_model(dataset_sample, load_param='params_ebmdiff_gmm.p', ebm=True)
        spiral_params_diff = train_single_model(dataset_sample, load_param='params_diff_gmm.p', ebm=False)

        # Load Data and Train Model - Energy-Based Diffusion Model and Diffusion Model
        nll_bar, dataset_sample = bar(scale=0.2)
        bar_params_ebm = train_single_model(dataset_sample, load_param='params_ebmdiff_bar.p', ebm=True)
        bar_params_diff = train_single_model(dataset_sample, load_param='params_diff_bar.p', ebm=False)

        # Collect Params
        params_ebm = {}
        for k, v in spiral_params_ebm.items():
            params_ebm[k] = v

        for k, v in bar_params_ebm.items():
            k = k.replace('resnet_diffusion_model/', 'resnet_diffusion_model_1/')
            params_ebm[k] = v

        # Collect Params Diff
        params_diff = {}
        for k, v in spiral_params_diff.items():
            params_diff[k] = v

        for k, v in bar_params_diff.items():
            k = k.replace('resnet_diffusion_model/', 'resnet_diffusion_model_1/')
            params_diff[k] = v

        # Sample with Product Distribution
        # samples, grad_sample, _ = sampling_product_distribution(params_ebm, ebm=True, grad=True)
        # print('LL EBM: ', np.mean(-nll_gmm(samples) - nll_bar(samples)))
        # print('LL Reverse: ', np.mean(-nll_gmm(grad_sample) - nll_bar(grad_sample)))
        samples, grad_sample, _ = sampling_product_distribution(params_diff, ebm=False, grad=True)
        print('LL Reverse: ', np.mean(-nll_gmm(grad_sample) - nll_bar(grad_sample)))


        # Sample with Product Distribution
        samples, _, _ = sampling_product_distribution(params_diff, ebm=False, n_trapets=0)
        print('LL Diff Taylor ', np.mean(-nll_gmm(samples) - nll_bar(samples)))

        # Sample with Product Distribution
        samples, _, _ = sampling_product_distribution(params_diff, ebm=False, n_trapets=2)
        print('LL Diff Trapets-2 ', np.mean(-nll_gmm(samples) - nll_bar(samples)))

        # Sample with Product Distribution
        samples, _, _ = sampling_product_distribution(params_diff, ebm=False, n_trapets=5)
        print('LL Diff Trapets-5 ', np.mean(-nll_gmm(samples) - nll_bar(samples)))

        # Sample with Product Distribution
        samples, _, _ = sampling_product_distribution(params_diff, ebm=False, n_trapets=10)
        print('LL Diff Trapets-10 ', np.mean(-nll_gmm(samples) - nll_bar(samples)))

        # Sample with Product Distribution
        samples, _, _ = sampling_product_distribution(params_diff, ebm=False, n_trapets=50)
        print('LL Diff Trapets-50', np.mean(-nll_gmm(samples) - nll_bar(samples)))
