from typing import Tuple
from collections.abc import Callable
import torch as th
import torch.nn as nn
import pytorch_lightning as pl
from src.model.base import EnergyModel


class DiffusionModel(pl.LightningModule):
    """Trainer for learning an unconditional diffusion model with fixed posterior variance"""

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

        self.require_g = False
        if isinstance(self.model, EnergyModel):
            self.require_g = True

    def _process_batch(self, batch) -> Tuple[int, th.Tensor]:
        """Hack to handle multiple formats of dataloaders"""
        if isinstance(batch, dict):
            batch_size = batch["pixel_values"].size(0)
            x = batch["pixel_values"].to(self.device)
        else:
            # Discard label
            x, _ = batch
            x = x.to(self.device)
            batch_size = x.size(0)
        return batch_size, x

    def training_step(self, batch, batch_idx):
        batch_size, x = self._process_batch(batch)
        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        T = self.noise_scheduler.time_steps.size(0)
        ts = th.randint(0, T, (batch_size,), device=self.device).long()

        noise = th.randn_like(x)
        x_noisy = self.noise_scheduler.q_sample(x_0=x, ts=ts, noise=noise)
        if self.require_g:
            x_noisy = x_noisy.requires_grad_(True)
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
        th.set_grad_enabled(True)
        batch_size, x = self._process_batch(batch)

        rng_state = th.get_rng_state()
        th.manual_seed(self.i_batch_val)

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        T = self.noise_scheduler.time_steps.size(0)
        ts = th.randint(0, T, (batch_size,), device=self.device).long()

        noise = th.randn_like(x)
        th.set_rng_state(rng_state)

        x_noisy = self.noise_scheduler.q_sample(x_0=x, ts=ts, noise=noise)
        if self.require_g:
            x_noisy = x_noisy.requires_grad_(True)
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


class LearnedVarDiffusion(pl.LightningModule):
    """Trainer for learning an unconditional diffusion model with learned posterior variance"""

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

        self.require_g = False
        if isinstance(self.model, EnergyModel):
            self.require_g = True

    def training_step(self, batch, batch_idx):
        batch_size = batch["pixel_values"].size(0)
        x = batch["pixel_values"].to(self.device)

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        T = self.noise_scheduler.time_steps.size(0)
        ts = th.randint(0, T, (batch_size,), device=self.device).long()

        noise = th.randn_like(x)
        x_noisy = self.noise_scheduler.q_sample(x_0=x, ts=ts, noise=noise)
        if self.require_g:
            x_noisy = x_noisy.requires_grad_(True)
        predicted_noise = self._predicted_noise(x_noisy, ts)

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
        th.set_grad_enabled(True)
        batch_size = batch["pixel_values"].size(0)
        x = batch["pixel_values"].to(self.device)

        rng_state = th.get_rng_state()
        th.manual_seed(self.i_batch_val)

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        T = self.noise_scheduler.time_steps.size(0)
        ts = th.randint(0, T, (batch_size,), device=self.device).long()

        noise = th.randn_like(x)
        th.set_rng_state(rng_state)

        x_noisy = self.noise_scheduler.q_sample(x_0=x, ts=ts, noise=noise)
        if self.require_g:
            x_noisy = x_noisy.requires_grad_(True)

        predicted_noise = self._predicted_noise(x_noisy, ts)
        loss = self.loss_f(noise, predicted_noise)
        self.log("val_loss", loss)
        self.val_loss += loss.detach().cpu().item()
        self.i_batch_val += 1
        return loss

    def _predicted_noise(self, x_noisy, ts):
        """Ignore estimated variance and return only predicted noise"""
        # Model returns eps_thet, post var estimate
        output = self.model(x_noisy, ts)
        # Split output tensor (B, 2*channels, img_size, img_size)
        # into eps_theta (B, channels, img_size, img_size) and discard var. estimate.
        predicted_noise, _ = output.split(x_noisy.size(1), dim=1)
        return predicted_noise

    def on_validation_epoch_end(self):
        print(" {}. Validation Loss: {}".format(self.i_epoch, self.val_loss / self.i_batch_val))
        self.val_loss = 0.0
        self.i_batch_val = 0


class DiffusionClassifier(pl.LightningModule):
    """Trainer for learning classification model p(y | x_t),
    where x_t is a noisy sample from a forward diffusion process."""

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
        batch_size = batch["pixel_values"].size(0)
        x = batch["pixel_values"].to(self.device).float()
        y = batch["label"].to(self.device).long()

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        T = self.noise_scheduler.time_steps.size(0)
        ts = th.randint(0, T, (batch_size,), device=self.device).long()

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
        batch_size = batch["pixel_values"].size(0)
        x = batch["pixel_values"].to(self.device)
        y = batch["label"].to(self.device)

        rng_state = th.get_rng_state()
        th.manual_seed(self.i_batch_val)

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        T = self.noise_scheduler.time_steps.size(0)
        ts = th.randint(0, T, (batch_size,), device=self.device).long()

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
