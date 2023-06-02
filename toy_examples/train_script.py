import time
import math
import functools
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import haiku as hk
import distrax
import chex
import optax
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds

from toy_examples.models import (
    ResnetDiffusionModel,
    EBMDiffusionModel,
)
from toy_examples.train import PortableDiffusionModel, bar

# Use a EBM formulation of likelihod vs a score formulation of likelihood
ebm = True

# Number of diffusion timesteps to train
n_steps = 100
data_dim = 2

batch_size = 1000
num_steps = 15001

EMA = 0.999


net_params = {"n_layers": 4, "h_dim": 128, "emb_dim": 32}


def forward_fn():
    net = ResnetDiffusionModel(
        n_steps=n_steps, n_layers=4, x_dim=data_dim, h_dim=128, emb_dim=32
    )

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


forward = hk.multi_transform(forward_fn)
rng_seq = hk.PRNGSequence(0)

xr = [-0.75, 0.75]
yr = [-0.75, 0.75]


# load data
dataset_energy, dataset_sample = bar(scale=0.2)
x = dataset_sample(batch_size)
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
        logpx = logpx_fn(ema_params, next(rng_seq), x).mean()
        print("TEST", itr, "logpx", logpx)
        test_logpx.append(logpx)

bar_params = ema_params
