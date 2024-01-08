import torch.nn as nn
import torch.nn.functional as F


class ResnetDiffusionModel(nn.Module):
    """Resnet score model.

    Adds embedding for each scale after each linear layer.
    """

    def __init__(
        self,
        num_diff_steps,
        n_layers=4,
        x_dim=2,
        h_dim=128,
        emb_dim=32,
        widen=2,
        emb_type="learned",
    ):
        assert emb_type in ("learned", "sinusoidal")
        super().__init__()
        self._n_steps = num_diff_steps
        self._n_layers = n_layers
        self._x_dim = x_dim
        self._h_dim = h_dim
        self._emb_dim = emb_dim
        self._widen = widen

        # Modules
        if emb_type == "learned":
            self.emb = nn.Embedding(self._n_steps, self._emb_dim)
        else:
            self.emb = partial(timestep_embedding, dim=self._emb_dim)
        self.first = nn.Linear(self._x_dim, self._h_dim)

        self.blocks = nn.ModuleList([])
        for _ in range(self._n_layers):
            self.blocks.append(Block(self._h_dim, self._emb_dim, self._widen))

        self.out = nn.Linear(self._h_dim, self._x_dim)

    def forward(self, x, t):
        emb = self.emb(t)
        x = self.first(x)
        for block in self.blocks:
            x = block(x, emb)
        x = self.out(x)
        return x


class Block(nn.Module):
    def __init__(self, input_dim, time_emb_dim, widen):
        super().__init__()
        wide_dim = input_dim * widen
        self.layer_h = nn.Linear(input_dim, wide_dim)
        self.layer_emb = nn.Linear(time_emb_dim, wide_dim)
        self.layer_int = nn.Linear(wide_dim, wide_dim)
        # TODO: NB initialise below layer to zero in R3
        self.layer_out = nn.Linear(wide_dim, input_dim)

        # TODO: Compare layer norms
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, h_in, time_emb):
        h_out = self.norm(h_in)
        h_out = F.silu(h_out)
        h_out = self.layer_h(h_out)
        h_out = h_out + self.layer_emb(time_emb)
        h_out = F.silu(h_out)
        h_out = self.layer_int(h_out)
        h_out = F.silu(h_out)
        h_out = self.layer_out(h_out)
        return h_in + h_out


# class JaxResnetDiffusionModel(hk.Module):
#     """Resnet score model.
#
#     Adds embedding for each scale after each linear layer.
#     """
#
#     def __init__(
#         self,
#         n_steps,
#         n_layers,
#         x_dim,
#         h_dim,
#         emb_dim,
#         widen=2,
#         emb_type="learned",
#         name=None,
#     ):
#         assert emb_type in ("learned", "sinusoidal")
#         super().__init__(name=name)
#         self._n_layers = n_layers
#         self._n_steps = n_steps
#         self._x_dim = x_dim
#         self._h_dim = h_dim
#         self._emb_dim = emb_dim
#         self._widen = widen
#         self._emb_type = emb_type
#
#     def __call__(self, x, t):
#         x = jnp.atleast_2d(x)
#         t = jnp.atleast_1d(t)
#
#         chex.assert_shape(x, (None, self._x_dim))
#         chex.assert_shape(t, (None,))
#         chex.assert_type([x, t], [jnp.float32, jnp.int64])
#
#         if self._emb_type == "learned":
#             emb = hk.Embed(self._n_steps, self._emb_dim)(t)
#         else:
#             emb = timestep_embedding(t, self._emb_dim)
#
#         x = hk.Linear(self._h_dim)(x)
#
#         for _ in range(self._n_layers):
#             # get layers and embeddings
#             layer_h = hk.Linear(self._h_dim * self._widen)
#             layer_emb = hk.Linear(self._h_dim * self._widen)
#             layer_int = hk.Linear(self._h_dim * self._widen)
#             layer_out = hk.Linear(self._h_dim, w_init=jnp.zeros)
#
#             h = hk.LayerNorm(-1, True, True)(x)
#             h = jax.nn.swish(h)
#             h = layer_h(h)
#             h += layer_emb(emb)
#             h = jax.nn.swish(h)
#             h = layer_int(h)
#             h = jax.nn.swish(h)
#             h = layer_out(h)
#             x += h
#
#         x = hk.Linear(self._x_dim, w_init=jnp.zeros)(x)
#         chex.assert_shape(x, (None, self._x_dim))
#         return x
