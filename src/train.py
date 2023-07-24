import numpy as np

import jax
import jax.numpy as jnp
import haiku as hk
import distrax
import chex

from src.utils import cosine_beta_schedule, extract


def bar(scale=0.2):
    """Ring of 2D Gaussians. Returns energy and sample functions."""

    def nll(_x):
        return 1

    def sample(n_samples):
        data = np.random.uniform(-1, 1, (n_samples, 2))
        data[:, 0] = data[:, 0] * scale

        return data

    return nll, sample


def toy_gmm(n_comp=8, std=0.075, radius=0.5):
    """Ring of 2D Gaussians. Returns energy and sample functions."""

    means_x = np.cos(2 * np.pi * np.linspace(0, (n_comp - 1) / n_comp, n_comp)).reshape(
        n_comp, 1, 1, 1
    )
    means_y = np.sin(2 * np.pi * np.linspace(0, (n_comp - 1) / n_comp, n_comp)).reshape(
        n_comp, 1, 1, 1
    )
    mean = radius * np.concatenate((means_x, means_y), axis=1)
    weights = np.ones(n_comp) / n_comp

    def nll(x):
        means = jnp.array(mean.reshape((-1, 1, 2)))
        c = np.log(n_comp * 2 * np.pi * std ** 2)
        f = (
            jax.nn.logsumexp(
                jnp.sum(-0.5 * jnp.square((x - means) / std), axis=2), axis=0
            )
            + c
        )
        # f = f + np.log(2)

        return f

    def sample(n_samples):
        toy_sample = np.zeros(0).reshape((0, 2, 1, 1))
        sample_group_sz = np.random.multinomial(n_samples, weights)
        for i in range(n_comp):
            sample_group = mean[i] + std * np.random.randn(
                2 * sample_group_sz[i]
            ).reshape(-1, 2, 1, 1)
            toy_sample = np.concatenate((toy_sample, sample_group), axis=0)
            np.random.shuffle(toy_sample)
        data = toy_sample[:, :, 0, 0]

        return data

    return nll, sample


# Simple diffusion model training
class PortableDiffusionModel(hk.Module):
    """Basic Diffusion Model."""

    def __init__(
        self,
        dim,
        n_steps,
        net,
        loss_type="simple",
        mc_loss=True,
        var_type="learned",
        samples_per_step=1,
        name=None,
    ):
        super().__init__(name=name)
        assert var_type in ("beta_forward", "beta_reverse", "learned")
        self._var_type = var_type
        self.net = net
        self._n_steps = n_steps
        self._dim = dim
        self._loss_type = loss_type
        self._mc_loss = mc_loss
        self._samples_per_step = samples_per_step
        self._betas = cosine_beta_schedule(n_steps)

        self._alphas = 1.0 - self._betas
        self._log_alphas = jnp.log(self._alphas)

        alphas = 1.0 - self._betas

        self._sqrt_alphas = jnp.array(jnp.sqrt(alphas), dtype=jnp.float32)
        self._sqrt_recip_alphas = jnp.array(1.0 / jnp.sqrt(alphas), dtype=jnp.float32)

        self._alphas_cumprod = jnp.cumprod(self._alphas, axis=0)
        self._alphas_cumprod_prev = jnp.append(1.0, self._alphas_cumprod[:-1])
        self._sqrt_alphas_cumprod = jnp.sqrt(self._alphas_cumprod)
        self._sqrt_one_minus_alphas_cumprod = jnp.sqrt(1 - self._alphas_cumprod)
        self._log_one_minus_alphas_cumprod = jnp.log(1 - self._alphas_cumprod)

        self._sqrt_recip_alphas_cumprod = jax.lax.rsqrt(self._alphas_cumprod)
        self._sqrt_recipm1_alphas_cumprod = jnp.sqrt(1 / self._alphas_cumprod - 1)
        self._sqrt_recipm1_alphas_cumprod_custom = jnp.sqrt(
            1.0 / (1 - self._alphas_cumprod)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self._posterior_variance = (
            self._betas
            * (1.0 - self._alphas_cumprod_prev)
            / (1.0 - self._alphas_cumprod)
        )

        self._posterior_log_variance_clipped = jnp.log(
            jnp.clip(self._posterior_variance, a_min=jnp.min(self._betas))
        )
        self._posterior_mean_coef1 = (
            self._betas
            * jnp.sqrt(self._alphas_cumprod_prev)
            / (1 - self._alphas_cumprod)
        )
        self._posterior_mean_coef2 = (
            (1 - self._alphas_cumprod_prev)
            * jnp.sqrt(self._alphas)
            / (1 - self._alphas_cumprod)
        )

        self._out_logvar = hk.get_parameter(
            "out_logvar",
            shape=(n_steps,),
            init=hk.initializers.Constant(jnp.log(self._betas)),
        )

    def energy_scale(self, t):
        return self._sqrt_recipm1_alphas_cumprod[t]

    def data_scale(self, t):
        return self._sqrt_recip_alphas_cumprod[t]

    def forward(self, x, t):
        """Get mu_t-1 given x_t."""
        x = jnp.atleast_2d(x)
        t = jnp.atleast_1d(t)

        chex.assert_shape(x, (None, self._dim))
        chex.assert_shape(t, (None,))
        chex.assert_type(t, jnp.int64)

        outs = self.net(x, t)
        chex.assert_shape(outs, x.shape)
        return outs

    def stats(self):
        """Returns static variables for computing variances."""
        return {
            "betas": self._betas,
            "alphas": self._alphas,
            "alphas_cumprod": self._alphas_cumprod,
            "alphas_cumprod_prev": self._alphas_cumprod_prev,
            "sqrt_alphas_cumprod": self._sqrt_alphas_cumprod,
            "sqrt_one_minus_alphas_cumprod": self._sqrt_one_minus_alphas_cumprod,
            "log_one_minus_alphas_cumprod": self._log_one_minus_alphas_cumprod,
            "sqrt_recip_alphas_cumprod": self._sqrt_recip_alphas_cumprod,
            "sqrt_recipm1_alphas_cumprod": self._sqrt_recipm1_alphas_cumprod,
            "posterior_variance": self._posterior_variance,
            "posterior_log_variace_clipped": self._posterior_log_variance_clipped,
        }

    def q_mean_variance(self, x_0, t):
        """Returns parameters of q(x_t | x_0)."""
        mean = extract(self._sqrt_alphas_cumprod, t, x_0.shape) * x_0
        variance = extract(1.0 - self._alphas_cumprod, t, x_0.shape)
        log_variance = extract(self._log_one_minus_alphas_cumprod, t, x_0.shape)
        return mean, variance, log_variance

    def q_sample(self, x_0, t, noise=None):
        """Sample from q(x_t | x_0)."""
        chex.assert_shape(x_0, (None, self._dim))
        chex.assert_shape(t, (None,))
        chex.assert_type([x_0, t], [jnp.float32, jnp.int64])

        if noise is None:
            noise = jax.random.normal(hk.next_rng_key(), x_0.shape)

        x_t = (
            extract(self._sqrt_alphas_cumprod, t, x_0.shape) * x_0
            + extract(self._sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise
        )
        chex.assert_shape(x_t, x_0.shape)
        return x_t

    def p_loss_simple(self, x_0, t):
        """Training loss for given x_0 and t."""
        chex.assert_shape(x_0, (None, self._dim))
        chex.assert_shape(t, (None,))
        chex.assert_type([x_0, t], [jnp.float32, jnp.int64])

        noise = jax.random.normal(hk.next_rng_key(), x_0.shape)
        x_noise = self.q_sample(x_0, t, noise)
        noise_recon = self.forward(x_noise, t)
        chex.assert_shape(noise_recon, x_0.shape)
        mse = jnp.square(noise_recon - noise)

        chex.assert_shape(mse, (t.shape[0], self._dim))
        mse = jnp.mean(mse, axis=1)  # avg over the output dimension

        chex.assert_shape(mse, t.shape)
        return mse

    def p_loss_kl(self, x_0, t):
        """Training loss for given x_0 and t (KL-weighted)."""
        chex.assert_shape(x_0, (None, self._dim))
        chex.assert_shape(t, (None,))
        chex.assert_type([x_0, t], [jnp.float32, jnp.int64])

        x_t = self.q_sample(x_0, t)
        q_mean, _, q_log_variance = self.q_posterior(x_0, x_t, t)
        p_mean, _, p_log_variance = self.p_mean_variance(x_t, t)

        dist_q = distrax.Normal(q_mean, jnp.exp(0.5 * q_log_variance))

        def _loss(pmu, plogvar):
            dist_p = distrax.Normal(pmu, jnp.exp(0.5 * plogvar))
            kl = dist_q.kl_divergence(dist_p).mean(-1)
            nll = -dist_p.log_prob(x_0).mean(-1)
            return kl, nll, jnp.where(t == 0, nll, kl)

        kl, nll, loss = _loss(p_mean, p_log_variance)

        chex.assert_equal_shape([nll, kl])
        chex.assert_shape(loss, (t.shape[0],))
        return loss

    def q_posterior(self, x_0, x_t, t):
        """Obtain parameters of q(x_{t-1} | x_0, x_t)."""
        chex.assert_shape(x_0, (None, self._dim))
        chex.assert_shape(t, (None,))
        chex.assert_type([x_0, t], [jnp.float32, jnp.int64])

        mean = (
            extract(self._posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(self._posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(self._posterior_variance, t, x_t.shape)
        log_var_clipped = extract(self._posterior_log_variance_clipped, t, x_t.shape)
        chex.assert_equal_shape([var, log_var_clipped])
        chex.assert_equal_shape([x_0, x_t, mean])
        return mean, var, log_var_clipped

    def predict_start_from_noise(self, x_t, t, noise):
        """Predict x_0 from x_t."""
        chex.assert_shape(x_t, (None, self._dim))
        chex.assert_shape(t, (None,))
        chex.assert_type([x_t, t], [jnp.float32, jnp.int64])

        x_0 = (
            extract(self._sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self._sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
        chex.assert_shape(x_0, x_t.shape)
        return x_0

    def p_mean_variance(self, x, t, clip=jnp.inf):
        """Parameters of p(x_{t-1} | x_t)."""
        chex.assert_shape(x, (None, self._dim))
        chex.assert_shape(t, (None,))
        chex.assert_type([x, t], [jnp.float32, jnp.int64])

        x_recon = jnp.clip(
            self.predict_start_from_noise(x, t, noise=self.forward(x, t)), -clip, clip
        )

        mean, var, log_var = self.q_posterior(x_recon, x, t)

        chex.assert_shape(var, (x.shape[0], 1))
        chex.assert_equal_shape([var, log_var])
        chex.assert_shape(mean, x.shape)

        if self._var_type == "beta_reverse":
            pass
        elif self._var_type == "beta_forward":
            var = extract(self._betas, t, x.shape)
            log_var = jnp.log(var)
        elif self._var_type == "learned":
            log_var = extract(self._out_logvar, t, x.shape)
            var = jnp.exp(log_var)
        else:
            raise ValueError(f"{self._var_type} not recognised.")

        chex.assert_shape(var, (x.shape[0], 1))
        chex.assert_equal_shape([var, log_var])
        chex.assert_shape(mean, (x.shape[0], x.shape[1]))
        return mean, var, log_var

    def p_sample(self, x, t, rng_key=None, clip=jnp.inf):
        """Sample from p(x_{t-1} | x_t)."""
        chex.assert_shape(x, (None, self._dim))
        chex.assert_shape(t, (None,))
        chex.assert_type([x, t], [jnp.float32, jnp.int64])

        mean, _, log_var = self.p_mean_variance(x, t, clip=clip)

        if rng_key is None:
            rng_key = hk.next_rng_key()
        noise = jax.random.normal(rng_key, x.shape)

        x_tm1 = mean + jnp.exp(0.5 * log_var) * noise
        chex.assert_equal_shape([x, x_tm1])
        return x_tm1

    def _prior_kl(self, x_0):
        """KL(q_T(x) || p(x))."""
        t = jnp.ones((x_0.shape[0],), dtype=jnp.int64) * (self._n_steps - 1)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_0, t)
        qt_dist = distrax.Normal(qt_mean, jnp.exp(0.5 * qt_log_variance))
        p_dist = distrax.Normal(jnp.zeros_like(qt_mean), jnp.ones_like(qt_mean))
        kl = qt_dist.kl_divergence(p_dist).mean(-1)
        chex.assert_shape(kl, (x_0.shape[0],))
        return kl

    def logpx(self, x_0):
        """Full elbo estimate of model."""
        e = self._prior_kl(x_0)
        chex.assert_shape(e, (x_0.shape[0],))
        n_repeats = self._n_steps * self._samples_per_step
        e = e.repeat(n_repeats, axis=0) / n_repeats

        kls = self.loss_all_t(x_0, loss_type="kl")
        logpx = -(kls + e) * self._dim * self._n_steps
        return {"logpx": logpx}

    def sample(self, n, clip=jnp.inf):
        """Sample from p(x)."""
        chex.assert_type(n, int)
        rng_key = hk.next_rng_key()
        rng_key, r = jax.random.split(rng_key)

        x = jax.random.normal(r, (n, self._dim))

        def body_fn(i, inputs):
            rng_key, x = inputs
            rng_key, r = jax.random.split(rng_key)
            j = self._n_steps - 1 - i
            t = jnp.ones((n,), dtype=jnp.int64) * j
            x = self.p_sample(x, t, rng_key=r, clip=clip)
            return rng_key, x

        x = hk.fori_loop(0, self._n_steps, body_fn, (rng_key, x))[1]

        chex.assert_shape(x, (n, self._dim))
        return x

    def loss(self, x):
        if self._mc_loss:
            return self.loss_mc(x, loss_type=self._loss_type)
        else:
            return self.loss_all_t(x, loss_type=self._loss_type)

    def loss_mc(self, x, loss_type=None):
        """Compute training loss, uniformly sampling t's."""
        chex.assert_shape(x, (None, self._dim))

        t = jax.random.randint(hk.next_rng_key(), (x.shape[0],), 0, self._n_steps)
        if loss_type == "simple":
            loss = self.p_loss_simple(x, t)
            # loss = self.p_loss_simple_cv(x, t)
        elif loss_type == "kl":
            loss = self.p_loss_kl(x, t)
        else:
            raise ValueError(f"Unrecognized loss type: {loss_type}")

        chex.assert_shape(loss, (x.shape[0],))
        return loss

    def loss_all_t(self, x, loss_type=None):
        """Compute training loss enumerated and averaged over all t's."""
        chex.assert_shape(x, (None, self._dim))
        x = jnp.array(x)
        t = jnp.concatenate([jnp.arange(0, self._n_steps)] * x.shape[0])
        t = jnp.tile(t[None], (self._samples_per_step,)).reshape(-1)
        x_r = jnp.tile(x[None], (self._n_steps * self._samples_per_step,)).reshape(
            -1, *x.shape[1:]
        )
        chex.assert_equal_shape_prefix((x_r, t), 1)

        if loss_type == "simple":
            loss = self.p_loss_simple(x_r, t)
        elif loss_type == "kl":
            loss = self.p_loss_kl(x_r, t)
        else:
            raise ValueError(f"Unrecognized loss type: {loss_type}")
        return loss

    def p_gradient(self, x, t, clip=jnp.inf):
        """Compute mean and variance of Gaussian reverse model p(x_{t-1} | x_t)."""
        b = x.shape[0]
        # chex.assert_axis_dimension(t, 0, b)
        gradient = self.forward(x, t)
        gradient = gradient * extract(
            self._sqrt_recipm1_alphas_cumprod_custom, t, gradient.shape
        )

        return gradient

    def p_energy(self, x, t, clip=jnp.inf):
        """Compute mean and variance of Gaussian reverse model p(x_{t-1} | x_t)."""
        b = x.shape[0]
        # chex.assert_axis_dimension(t, 0, b)

        x = jnp.atleast_2d(x)
        t = jnp.atleast_1d(t)

        chex.assert_shape(x, (None, self._dim))
        chex.assert_shape(t, (None,))
        chex.assert_type(t, jnp.int64)

        energy = self.net.neg_logp_unnorm(x, t)
        energy = energy * extract(
            self._sqrt_recipm1_alphas_cumprod_custom, t, energy.shape
        )

        return energy
