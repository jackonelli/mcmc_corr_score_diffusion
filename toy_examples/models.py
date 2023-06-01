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


# Define a energy diffusion model (wrapper around a normal diffusion model)
class EBMDiffusionModel(hk.Module):
    """EBM parameterization on top of score model.

    Adds embedding for each scale after each linear layer.
    """

    def __init__(self, net, name=None):
        super().__init__(name=name)
        self.net = net

    def neg_logp_unnorm(self, x, t):
        score = self.net(x, t)
        return ((score - x) ** 2).sum(-1)

    def __call__(self, x, t):
        neg_logp_unnorm = lambda _x: self.neg_logp_unnorm(_x, t).sum()
        return hk.grad(neg_logp_unnorm)(x)


# Define how to multiply two different EBM distributions together
class ProductEBMDiffusionModel(hk.Module):
    """EBM where we compose two distributions together.

    Add the energy value together
    """

    def __init__(self, net, net2, name=None):
        super().__init__(name=name)
        self.net = net
        self.net2 = net2

    def neg_logp_unnorm(self, x, t):
        unorm_1 = self.net.neg_logp_unnorm(x, t)
        unorm_2 = self.net2.neg_logp_unnorm(x, t)
        return unorm_1 + unorm_2

    def __call__(self, x, t):
        score = self.net(x, t) + self.net2(x, t)
        return score


# Define a simple MLP Diffusion Model
class ResnetDiffusionModel(hk.Module):
    """Resnet score model.

    Adds embedding for each scale after each linear layer.
    """

    def __init__(
        self,
        n_steps,
        n_layers,
        x_dim,
        h_dim,
        emb_dim,
        widen=2,
        emb_type="learned",
        name=None,
    ):
        assert emb_type in ("learned", "sinusoidal")
        super().__init__(name=name)
        self._n_layers = n_layers
        self._n_steps = n_steps
        self._x_dim = x_dim
        self._h_dim = h_dim
        self._emb_dim = emb_dim
        self._widen = widen
        self._emb_type = emb_type

    def __call__(self, x, t):

        x = jnp.atleast_2d(x)
        t = jnp.atleast_1d(t)

        chex.assert_shape(x, (None, self._x_dim))
        chex.assert_shape(t, (None,))
        chex.assert_type([x, t], [jnp.float32, jnp.int64])

        if self._emb_type == "learned":
            emb = hk.Embed(self._n_steps, self._emb_dim)(t)
        else:
            emb = timestep_embedding(t, self._emb_dim)

        x = hk.Linear(self._h_dim)(x)

        for _ in range(self._n_layers):
            # get layers and embeddings
            layer_h = hk.Linear(self._h_dim * self._widen)
            layer_emb = hk.Linear(self._h_dim * self._widen)
            layer_int = hk.Linear(self._h_dim * self._widen)
            layer_out = hk.Linear(self._h_dim, w_init=jnp.zeros)

            h = hk.LayerNorm(-1, True, True)(x)
            h = jax.nn.swish(h)
            h = layer_h(h)
            h += layer_emb(emb)
            h = jax.nn.swish(h)
            h = layer_int(h)
            h = jax.nn.swish(h)
            h = layer_out(h)
            x += h

        x = hk.Linear(self._x_dim, w_init=jnp.zeros)(x)
        chex.assert_shape(x, (None, self._x_dim))
        return x
