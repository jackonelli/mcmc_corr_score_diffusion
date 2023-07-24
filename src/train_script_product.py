"""2D product composition experiments

The script trains two diffusion models, one score parameterised and one energy parameterised,
or loads existing parameters.
It then samples from these models with various MCMC methods.
"""
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from random import randint
import time
import pickle
import jax
import jax.numpy as jnp
import haiku as hk
import distrax
import numpy as np
import optax
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

from src.datasets import toy_gmm, bar
from src.models import (
    ResnetDiffusionModel,
    EBMDiffusionModel,
    PortableDiffusionModel,
    ProductEBMDiffusionModel,
)
from src.sampler import (
    AnnealedMUHASampler,
    AnnealedMUHADiffSampler,
    AnnealedULASampler,
    AnnealedUHASampler,
    AnnealedMALASampler,
    AnnealedMALADiffSampler,
    AnnealedMUHADiffReuseSampler,
)


def main():
    args = parse_args()
    # Parameter for Datasets
    n_comp = 8
    std = 0.03
    scale = 0.2
    r = 1.1
    prob_inside = 0.99
    n = 2000  # Number of samples from target distribution
    batch_size = 2000  # Number of generated samples from model
    if not args.exp_name.exists():
        args.exp_name.mkdir(parents=True)

    # Load Data
    # Gaussian Mixture
    _, dataset_sample_gmm, means = toy_gmm(n_comp, std=std)
    bounds_outer = jnp.array([[-r, r], [-r, r]])
    bounds_inner = jnp.array([[-scale, scale], [-1.0, 1.0]])

    # Bar
    _, dataset_sample_bar, pdf_outer, pdf_inner = bar(
        scale=scale, r=r, prob_inside=prob_inside
    )
    # c = compute_normalizing_constant(
    #     means, std, n_comp, pdf_outer, pdf_inner, bounds_outer, bounds_inner
    # )

    # Get models (train new model or load from file) - energy and score param
    print("Getting model params")
    for model_id in range(1, args.num_retrains + 1):
        print(f"Model {model_id}/{args.num_retrains}")
        param_path = args.exp_name / f"params_energy_gmm_{model_id}.p"
        gmm_params_ebm = train_single_model(
            dataset_sample_gmm,
            batch_size,
            args.num_training_steps,
            load_param=param_path if args.pre_trained else None,
            save_param=param_path,
            ebm=True,
            seed=args.seed,
        )
        param_path = args.exp_name / f"params_score_gmm_{model_id}.p"
        gmm_params_diff = train_single_model(
            dataset_sample_gmm,
            batch_size,
            args.num_training_steps,
            load_param=param_path if args.pre_trained else None,
            save_param=param_path,
            ebm=False,
            seed=args.seed,
        )

        param_path = args.exp_name / f"params_energy_bar_{model_id}.p"
        bar_params_ebm = train_single_model(
            dataset_sample_bar,
            batch_size,
            args.num_training_steps,
            load_param=param_path if args.pre_trained else None,
            save_param=param_path,
            ebm=True,
            seed=args.seed,
        )
        param_path = args.exp_name / f"params_score_bar_{model_id}.p"
        bar_params_diff = train_single_model(
            dataset_sample_bar,
            batch_size,
            args.num_training_steps,
            load_param=param_path if args.pre_trained else None,
            save_param=param_path,
            ebm=False,
            seed=args.seed,
        )

        # print("Sampling from diffusion models")

        # params_ebm = collect_product_params(gmm_params_ebm, bar_params_ebm)
        # params_diff = collect_product_params(gmm_params_diff, bar_params_diff)

        # samples_target = dataset_sample_gmm(n, bounds_inner[0], bounds_inner[1])
        # experiment_param = {
        #     "ebm_hmc": (params_ebm, True, "HMC", True, None),
        #     # "ebm_uhmc": (params_ebm, True, "UHMC", False, None),
        #     # "ebm_ula": (params_ebm, True, "ULA", False, None),
        #     # "ebm_mala": (params_ebm, True, "MALA", False, None),
        #     # "diff_hmc4eff": (params_diff, False, "effective", False, None),
        #     "diff_hmc3": (params_diff, False, "HMC", True, 3),
        #     # "diff_hmc5": (params_diff, False, "HMC", False, 5),
        #     # "diff_hmc10": (params_diff, False, "HMC", False, 10),
        #     # "diff_uhmc": (params_diff, False, "UHMC", False, None),
        #     # "diff_ula": (params_diff, False, "ULA", False, None),
        #     # "diff_mala3": (params_diff, False, "MALA", False, 3),
        #     # "diff_mala5": (params_diff, False, "MALA", False, 5),
        #     # "diff_mala10": (params_diff, False, "MALA", False, 10),
        # }

        # # results = dict()
        # samples_dict = dict()
        # samples_dict["target"] = samples_target
        # # samples_dict = pickle.load(open(file_samples, "rb"))

        # for name, param in experiment_param.items():
        #     print(f"Sampling with {name}")
        #     model_param, ebm, sampler, grad, n_trapets = param
        #     samples, grad_sample, _ = sampling_product_distribution(
        #         model_param,
        #         ebm=ebm,
        #         sampler=sampler,
        #         grad=grad,
        #         n_trapets=n_trapets,
        #         seed=args.seed,
        #     )
        #     samples_dict[name] = samples
        #     if grad:
        #         samples_dict[name.split("_")[0] + "_reverse"] = grad_sample

        #     file_samples = (
        #         args.exp_name / f"samples_{model_id}.p"
        #     )  # File to save samples
        #     print(f"Saving samples at {file_samples}")
        #     pickle.dump(samples_dict, open(file_samples, "wb"))


def train_single_model(
    dataset_sample,
    batch_size,
    num_training_steps,
    load_param=None,
    save_param=None,
    ebm=True,
    seed=None,
):
    if load_param is not None:
        print(f"Loading pre-trained model {load_param.name}")
        params = pickle.load(open(load_param, "rb"))
        return params

    partial_forward_fn = partial(forward_fn, ebm)

    forward = hk.multi_transform(partial_forward_fn)
    if seed is not None:
        rng_seq = hk.PRNGSequence(seed)
    else:
        rand_seed = randint(0, 9999)
        rng_seq = hk.PRNGSequence(jax.random.PRNGKey(rand_seed))

    x = dataset_sample(batch_size)

    # plot_samples(x)
    # plt.show()
    x = x.reshape(x.shape[0], -1)

    params = forward.init(next(rng_seq), x)
    loss_fn, sample_fn, logpx_fn, logp_unnorm_fn = forward.apply
    # param_count = sum(x.size for x in jax.tree_leaves(params))
    # for k, v in jax.tree_map(lambda x: x.shape, params).items():
    #     print(k, v)
    # print("Model has {} params".format(param_count))

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

    for itr in range(num_training_steps):
        x = dataset_sample(batch_size)

        x = x.reshape(x.shape[0], -1)
        start_time = time.time()
        loss, params, opt_state = update(params, opt_state, next(rng_seq), x)
        duration_update = time.time() - start_time
        ema_params = jax.tree_map(
            lambda e, p: e * EMA + p * (1 - EMA), ema_params, params
        )

        if itr % 100 == 0:
            print(itr, loss, "time:", duration_update)
            losses.append(loss)
        if itr % 1000 == 0:
            # x_samp = sample_fn(ema_params, next(rng_seq), batch_size)
            # plot_samples(x_samp)
            # plt.show()
            logpx = logpx_fn(ema_params, next(rng_seq), x).mean()
            print("TEST", itr, "logpx", logpx)
            test_logpx.append(logpx)

            # if ebm:
            #     dist_show_2d(
            #         lambda x: logp_unnorm_fn(ema_params, next(rng_seq), x, 0),
            #         xr=X_R,
            #         yr=Y_R,
            #     )
            #     plt.title(str(itr))
            #     plt.show()
            #     """
            #     for t in range(10):
            #         dist_show_2d(lambda x: logp_unnorm_fn(ema_params, next(rng_seq), x, 10 * t), xr=xr, yr=yr)
            #         plt.show()
            #     """
    # plt.plot(losses)
    # plt.show()
    # plt.plot(test_logpx)
    if save_param is not None:
        pickle.dump(ema_params, open(save_param, "wb"))
    return ema_params


def sampling_product_distribution(
    params, ebm=True, sampler="HMC", n_trapets=5, grad=False, batch_size=2000, seed=None
):
    partial_forward_fn_product = partial(forward_fn_product, ebm)
    forward_product = hk.multi_transform(partial_forward_fn_product)
    if seed is not None:
        rng_seq = hk.PRNGSequence(seed)
    else:
        rand_seed = randint(0, 9999)
        rng_seq = hk.PRNGSequence(jax.random.PRNGKey(rand_seed))

    if ebm:
        (
            _,
            dual_product_sample_fn,
            dual_product_nll,
            dual_product_logp_unorm_fn,
            dual_product_gradient_fn,
            dual_product_energy_fn,
        ) = forward_product.apply
    else:
        (
            _,
            dual_product_sample_fn,
            dual_product_nll,
            dual_product_logp_unorm_fn,
            dual_product_gradient_fn,
            dual_product_energy_fn,
        ) = forward_product.apply

    dual_product_sample_fn = jax.jit(dual_product_sample_fn, static_argnums=2)
    dual_product_gradient_fn = jax.jit(dual_product_gradient_fn)

    dual_product_energy_fn = jax.jit(dual_product_energy_fn)

    # Sampling from product of distributions
    dim = 2
    n_mode = 4
    std = 0.05
    init_std = 1.0
    init_mu = 0.0
    n_steps = 10
    damping = 0.5
    mass_diag_sqrt = 1.0
    num_leapfrog = 3
    samples_per_step = 10
    uha_step_size = 0.03
    ula_step_size = 0.001

    means = jax.random.normal(next(rng_seq), (n_mode, dim))
    comp_dists = distrax.MultivariateNormalDiag(means, jnp.ones_like(means) * std)
    pi = distrax.Categorical(logits=jnp.zeros((n_mode,)))
    target_dist = distrax.MixtureSameFamily(pi, comp_dists)
    initial_dist = distrax.MultivariateNormalDiag(
        means[0] * 0 + init_mu, init_std * jnp.ones_like(means[0])
    )

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

        betas = jnp.linspace(0.0, 1.0, n_steps)

        if sampler == "HMC":
            if ebm:
                sampler = AnnealedMUHASampler(
                    n_steps,
                    samples_per_step,
                    uha_step_sizes,
                    damping,
                    mass_diag_sqrt,
                    num_leapfrog,
                    initial_dist,
                    target_distribution=energy_function,
                    gradient_function=gradient_function,
                    energy_function=energy_function,
                )
            else:
                sampler = AnnealedMUHADiffSampler(
                    n_steps,
                    samples_per_step,
                    uha_step_sizes,
                    damping,
                    mass_diag_sqrt,
                    num_leapfrog,
                    initial_dist,
                    target_distribution=None,
                    gradient_function=gradient_function,
                    energy_function=energy_function,
                    n_trapets=n_trapets,
                )
        elif sampler == "MALA":
            if ebm:
                sampler = AnnealedMALASampler(
                    n_steps,
                    samples_per_step,
                    ula_step_sizes,
                    initial_dist,
                    target_distribution=None,
                    gradient_function=gradient_function,
                    energy_function=energy_function,
                )
            else:
                sampler = AnnealedMALADiffSampler(
                    n_steps,
                    samples_per_step,
                    ula_step_sizes,
                    initial_dist,
                    target_distribution=None,
                    gradient_function=gradient_function,
                    energy_function=energy_function,
                    n_trapets=n_trapets,
                )
        elif sampler == "UHMC":
            sampler = AnnealedUHASampler(
                n_steps,
                samples_per_step,
                uha_step_sizes,
                damping,
                mass_diag_sqrt,
                num_leapfrog,
                initial_dist,
                target_distribution=None,
                gradient_function=gradient_function,
            )
        elif sampler == "ULA":
            sampler = AnnealedULASampler(
                n_steps,
                samples_per_step,
                ula_step_sizes,
                initial_dist,
                target_distribution=None,
                gradient_function=gradient_function,
            )
        else:
            # raise ValueError('Not Valid Sampler Name')
            sampler = AnnealedMUHADiffReuseSampler(
                n_steps,
                samples_per_step,
                uha_step_sizes,
                damping,
                mass_diag_sqrt,
                num_leapfrog,
                initial_dist,
                target_distribution=None,
                gradient_function=gradient_function,
                energy_function=energy_function,
            )

        x_samp, logw, accept = sampler.sample(next(rng_seq), batch_size)
        # Samples from MCMC
        # plt.scatter(x_samp[:, 0], x_samp[:, 1], color="green", alpha=0.5)

        rng_seq = hk.PRNGSequence(seed)
        grad_sample = None
        if grad:
            grad_sample = dual_product_sample_fn(
                params, next(rng_seq), batch_size, jnp.inf
            )

            # Samples from adding score functions
            # plt.scatter(grad_sample[:, 0], grad_sample[:, 1], color="blue", alpha=0.5)

        # plt.show()

        return x_samp, grad_sample, accept


def forward_fn(ebm=True):
    net = ResnetDiffusionModel(
        n_steps=N_STEPS, n_layers=4, x_dim=DATA_DIM, h_dim=128, emb_dim=32
    )

    if ebm:
        net = EBMDiffusionModel(net)

    ddpm = PortableDiffusionModel(DATA_DIM, N_STEPS, net, var_type="beta_forward")

    def logp_unnorm(x, t):
        scale_e = ddpm.energy_scale(-2 - t)
        t = jnp.ones((x.shape[0],), dtype=jnp.int32) * t
        return -net.neg_logp_unnorm(x, t) * scale_e

    def _logpx(x):
        return ddpm.logpx(x)["logpx"]

    return ddpm.loss, (ddpm.loss, ddpm.sample, _logpx, logp_unnorm)


def forward_fn_product(ebm=True):
    net_one = ResnetDiffusionModel(
        n_steps=N_STEPS, n_layers=4, x_dim=DATA_DIM, h_dim=128, emb_dim=32
    )

    if ebm:
        net_one = EBMDiffusionModel(net_one)

    net_two = ResnetDiffusionModel(
        n_steps=N_STEPS, n_layers=4, x_dim=DATA_DIM, h_dim=128, emb_dim=32
    )

    if ebm:
        net_two = EBMDiffusionModel(net_two)

    dual_net = ProductEBMDiffusionModel(net_one, net_two)
    ddpm = PortableDiffusionModel(DATA_DIM, N_STEPS, dual_net, var_type="beta_forward")

    def logp_unnorm(x, t):
        scale_e = ddpm.energy_scale(-2 - t)
        t = jnp.ones((x.shape[0],), dtype=jnp.int32) * t
        return -dual_net.neg_logp_unnorm(x, t) * scale_e

    def _logpx(x):
        return ddpm.logpx(x)["logpx"]

    if ebm:
        return ddpm.loss, (
            ddpm.loss,
            ddpm.sample,
            _logpx,
            logp_unnorm,
            ddpm.p_gradient,
            ddpm.p_energy,
        )
    else:
        return ddpm.loss, (
            ddpm.loss,
            ddpm.sample,
            _logpx,
            logp_unnorm,
            ddpm.p_gradient,
            ddpm.p_gradient,
        )


def collect_product_params(gmm, bar):
    """Combine parameters for the two models into a composition"""
    params_comp = {}
    for k, v in gmm.items():
        params_comp[k] = v
    for k, v in bar.items():
        k = k.replace("resnet_diffusion_model/", "resnet_diffusion_model_1/")
        params_comp[k] = v
    return params_comp


def plot_samples(x):
    plt.scatter(x[:, 0], x[:, 1])
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)


def dist_show_2d(fn, xr, yr):
    nticks = 100
    x, y = np.meshgrid(
        np.linspace(xr[0], xr[1], nticks), np.linspace(yr[0], yr[1], nticks)
    )
    coord = np.stack([x, y], axis=-1).reshape((-1, 2))
    heatmap = fn(coord).reshape((nticks, nticks))
    plt.imshow(heatmap)


def parse_args():
    parser = ArgumentParser(prog="train_script_product")
    parser.add_argument(
        "--exp_name",
        default=None,
        type=Path,
        help="Directory to save parameters and samples to",
    )
    parser.add_argument(
        "--pre_trained", action="store_true", help="If set, loads parameters from file."
    )
    parser.add_argument(
        "--num_training_steps",
        default=15001,
        type=int,
        help="Number of training steps during training of the diff. models.",
    )
    parser.add_argument(
        "--num_retrains",
        default=1,
        type=int,
        help="Number of repetitions of the experiment.",
    )
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


DATA_DIM = 2
EMA = 0.999
N_STEPS = 100
NET_PARAMS = {"n_layers": 4, "h_dim": 128, "emb_dim": 32}

X_R = [-0.75, 0.75]
Y_R = [-0.75, 0.75]


if __name__ == "__main__":
    main()
