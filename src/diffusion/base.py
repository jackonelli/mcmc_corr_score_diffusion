"""Diffusion utils"""
from collections.abc import Callable
from abc import ABC
from typing import Union
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class DiffusionSampler(ABC):
    """Sampling from DDPM"""

    def __init__(
        self,
        beta_schedule: Callable,
        num_diff_steps: int,
        posterior_variance="beta",
    ):
        self.num_timesteps = num_diff_steps

        # define beta
        self.betas = beta_schedule(num_timesteps=num_diff_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = compute_alpha_bars(self.alphas)

        # Two different values of posterior variance (sigma ** 2) proposed of the authors
        if posterior_variance == "beta":
            self.posterior_variance = self.betas
        elif posterior_variance == "beta_tilde":
            self.posterior_variance = (
                self.betas * (1.0 - F.pad(self.alphas_bar[:-1], (1, 0), value=1.0)) / (1.0 - self.alphas_bar)
            )
        else:
            raise NotImplementedError

    def sigma_t(self, t, x_t):
        a_bar_t = extract(self.alphas_bar, t, x_t)
        return th.sqrt(1 - a_bar_t)

    def to(self, device: th.device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_bar = self.alphas_bar.to(device)
        self.posterior_variance = self.posterior_variance.to(device)

    def q_sample(self, x_0: th.Tensor, ts: th.Tensor, noise: th.Tensor):
        """
        Sampling from the forward process
        Add noise to the input tensor at random timesteps to produce a tensor of noisy samples

        Args:
            x_0 (any shape),
            ts (tensor of timesteps at which to add noise)

        Returns:
            noisy_samples (tensor of noisy samples of shape - x_0.shape)
        """

        x_t = _sample_x_t_given_x_0(x_0, ts, self.alphas_bar, noise)
        return x_t

    @th.no_grad()
    def sample(self, model: nn.Module, num_samples: int, device: th.device, shape: tuple):
        """Sampling from the backward process
        Sample points from the data distribution

        Args:
            model (model to predict noise)
            num_samples (number of samples)
            device (the device the model is on)
            shape (shape of data, e.g., (1, 28, 28))

        Returns:
            all x through the (predicted) reverse diffusion steps
        """

        steps = []
        x_tm1 = th.randn((num_samples,) + shape).to(device)

        for t in reversed(range(0, self.num_timesteps)):
            t_tensor = th.full((x_tm1.shape[0],), t, device=device)

            # Use the model to predict noise and use the noise to step back
            pred_noise = model(x_tm1, t_tensor)
            x_tm1 = self._sample_x_tm1_given_x_t(x_tm1, t, pred_noise)
            steps.append((t, x_tm1.detach().cpu()))

        return x_tm1, steps

    def _sample_x_tm1_given_x_t(self, x_t: th.Tensor, t: int, pred_noise: th.Tensor):
        """Denoise the input tensor at a given timestep using the predicted noise

        Args:
            x_t (any shape),
            t (timestep at which to denoise),
            predicted_noise (noise predicted at the timestep)

        Returns:
            x_tm1 (x[t-1] denoised sample by one step - x_t.shape)
        """

        b_t = extract(self.betas, t, x_t)
        a_t = extract(self.alphas, t, x_t)
        a_bar_t = extract(self.alphas_bar, t, x_t)
        post_var_t = extract(self.posterior_variance, t, x_t)

        if t > 0:
            z = th.randn_like(x_t)
        else:
            z = 0

        m_tm1 = (x_t - b_t / (th.sqrt(1 - a_bar_t)) * pred_noise) / a_t.sqrt()
        noise = post_var_t.sqrt() * z
        xtm1 = m_tm1 + noise
        return xtm1


def _sample_x_t_given_x_0(x_0: th.Tensor, ts: th.Tensor, alphas_bar: th.Tensor, noise: th.Tensor):
    """Sample from q(x_t | x_0)

    Add noise to the input tensor x_0 at given timesteps to produce a tensor of noisy samples x_t

    Args:
        x_0: [batch_size * [any shape]]
        ts: [batch_size,]
        alphas_bar [num_timesteps,]

    Returns:
        x_t: batch_size number of samples from x_t ~ q(x_t | x_0) [batch_size, *x.size()]
    """
    a_bar_t = extract(alphas_bar, ts, x_0)
    x_t = a_bar_t.sqrt() * x_0 + th.sqrt(1.0 - a_bar_t) * noise

    return x_t


def compute_alpha_bars(alphas):
    """Compute sequence of alpha_bar from sequence of alphas"""
    return th.cumprod(alphas, dim=0)


def extract(a: th.Tensor, t: Union[int, th.Tensor], x: th.Tensor):
    """Helper function to extract values of a tensor at given time steps

    Args:
        a,
        t,
        x

    Returns:
        out:
    """

    batch_size = x.shape[0]
    device = a.device
    inds = th.full((batch_size,), t).to(device) if isinstance(t, int) else t.to(device)
    out = a.gather(-1, inds)
    return out.reshape(batch_size, *((1,) * (len(x.shape) - 1))).to(x.device)


class DiffusionModel(pl.LightningModule):
    def __init__(self, model: nn.Module, loss_f: Callable, noise_scheduler):
        super().__init__()
        self.model = model
        self.loss_f = loss_f
        self.noise_scheduler = noise_scheduler

        # Default Initialization
        self.train_loss = 0.0
        self.val_loss = 0.0
        self.i_batch_train = 0
        self.i_batch_val = 0
        self.i_epoch = 0

    def training_step(self, batch, batch_idx):
        batch_size = batch["pixel_values"].shape[0]
        x = batch["pixel_values"].to(self.device)

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        ts = th.randint(0, self.noise_scheduler.num_timesteps, (batch_size,), device=self.device).long()

        noise = th.randn_like(x)
        x_noisy = self.noise_scheduler.q_sample(x_0=x, ts=ts, noise=noise)
        predicted_noise = self.model(x_noisy, ts)

        loss = self.loss_f(noise, predicted_noise)
        self.log("train_loss", loss)
        self.train_loss += loss.detach().cpu().item()
        self.i_batch_train += 1
        return loss

    def on_train_epoch_end(self):
        print(" {}. Train Loss: {}".format(self.i_epoch, self.train_loss / self.i_batch_train))
        self.train_loss = 0.0
        self.i_batch_train = 0
        self.i_epoch += 1

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = th.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):
        batch_size = batch["pixel_values"].shape[0]
        x = batch["pixel_values"].to(self.device)

        rng_state = th.get_rng_state()
        th.manual_seed(self.i_batch_val)

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        ts = th.randint(0, self.noise_scheduler.num_timesteps, (batch_size,), device=self.device).long()

        noise = th.randn_like(x)
        th.set_rng_state(rng_state)

        x_noisy = self.noise_scheduler.q_sample(x_0=x, ts=ts, noise=noise)
        predicted_noise = self.model(x_noisy, ts)

        loss = self.loss_f(noise, predicted_noise)
        self.log("val_loss", loss)
        self.val_loss += loss.detach().cpu().item()
        self.i_batch_val += 1
        return loss

    def on_validation_epoch_end(self):
        print(" {}. Validation Loss: {}".format(self.i_epoch, self.val_loss / self.i_batch_val))
        self.val_loss = 0.0
        self.i_batch_val = 0


class DiffusionClassifier(pl.LightningModule):
    def __init__(self, model: nn.Module, loss_f: Callable, noise_scheduler):
        super().__init__()
        self.model = model
        self.loss_f = loss_f
        self.noise_scheduler = noise_scheduler

        # Default Initialization
        self.train_loss = 0.0
        self.val_loss = 0.0
        self.i_batch_train = 0
        self.i_batch_val = 0
        self.i_epoch = 0

    def training_step(self, batch, batch_idx):
        batch_size = batch["pixel_values"].shape[0]
        x = batch["pixel_values"].to(self.device).float()
        y = batch["label"].to(self.device).long()

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        ts = th.randint(0, self.noise_scheduler.num_timesteps, (batch_size,), device=self.device).long()

        noise = th.randn_like(x)
        x_noisy = self.noise_scheduler.q_sample(x_0=x, ts=ts, noise=noise)
        predicted_y = self.model(x_noisy, ts)
        loss = self.loss_f(predicted_y, y)
        self.log("train_loss", loss)
        self.train_loss += loss.detach().cpu().item()
        self.i_batch_train += 1
        return loss

    def on_train_epoch_end(self):
        print(" {}. Train Loss: {}".format(self.i_epoch, self.train_loss / self.i_batch_train))
        self.train_loss = 0.0
        self.i_batch_train = 0
        self.i_epoch += 1

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = th.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):
        batch_size = batch["pixel_values"].shape[0]
        x = batch["pixel_values"].to(self.device)
        y = batch["label"].to(self.device)

        rng_state = th.get_rng_state()
        th.manual_seed(self.i_batch_val)

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        ts = th.randint(0, self.noise_scheduler.num_timesteps, (batch_size,), device=self.device).long()

        noise = th.randn_like(x)
        th.set_rng_state(rng_state)

        x_noisy = self.noise_scheduler.q_sample(x_0=x, ts=ts, noise=noise)
        predicted_y = self.model(x_noisy, ts)

        loss = self.loss_f(predicted_y, y)
        self.log("val_loss", loss)
        self.val_loss += loss.detach().cpu().item()
        self.i_batch_val += 1
        return loss

    def on_validation_epoch_end(self):
        print(" {}. Validation Loss: {}".format(self.i_epoch, self.val_loss / self.i_batch_val))
        self.val_loss = 0.0
        self.i_batch_val = 0
