from collections.abc import Callable
import torch as th
import torch.nn as nn
from src.model.base import EnergyModel
import pytorch_lightning as pl


class DiffusionModel(pl.LightningModule):
    def __init__(self, model: nn.Module, loss_f: Callable, noise_scheduler, fixed=False,
                 path_load_state=None, lr: float = 2e-4):
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

        if '_orig_mod' in dir(self.model):
            if isinstance(self.model._orig_mod, EnergyModel):
                self.require_g = True
        else:
            if isinstance(self.model, EnergyModel):
                self.require_g = True

        self.fixed_noise_t = {'noise': {}, 'ts': {}}
        self.fixed = fixed
        self.path_load_state = path_load_state
        self.lr = lr

    def training_step(self, batch, batch_idx):
        batch_size = batch["pixel_values"].shape[0]
        x = batch["pixel_values"].to(self.device)

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        ts = th.randint(0, self.noise_scheduler.num_diff_steps, (batch_size,), device=self.device).long()

        noise = th.randn_like(x)
        x_noisy = self.noise_scheduler.q_sample(x_0=x, ts=ts, noise=noise)
        if self.require_g:
            x_noisy = x_noisy.requires_grad_(True)
        predicted_noise = self.model(x_noisy, ts)

        loss = self.loss_f(noise, predicted_noise)
        self.train_loss += loss.detach().cpu().item()
        self.i_batch_train += 1
        self.log("train_loss", self.train_loss / self.i_batch_train)
        return loss

    def on_train_epoch_end(self):
        print(" {}. Train Loss: {}".format(self.i_epoch, self.train_loss / self.i_batch_train))
        self.train_loss = 0.0
        self.i_batch_train = 0
        self.i_epoch += 1

    def configure_optimizers(self):
        warmup = 5000
        def warmup_lr(step):
            return min(step, warmup) / warmup
        optimizer = th.optim.Adam(self.parameters(), lr=self.lr)

        checkpoint = None
        if self.path_load_state is not None:
            checkpoint = th.load(self.path_load_state)
            optimizer.load_state_dict(checkpoint['optimizer_states'][0])

        scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)
        if self.path_load_state is not None:
            scheduler.load_state_dict(checkpoint['lr_schedulers'][0])

        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
        },
    }

    def validation_step(self, batch, batch_idx):
        batch_size = batch["pixel_values"].shape[0]
        x = batch["pixel_values"].to(self.device)

        if self.fixed:
            if batch_idx in self.fixed_noise_t['noise']:
                noise = self.fixed_noise_t['noise'][batch_idx].to(self.device)
                ts = self.fixed_noise_t['ts'][batch_idx].to(self.device)
            else:
                ts = th.randint(0, self.noise_scheduler.num_diff_steps, (batch_size,), device=self.device).long()
                noise = th.randn_like(x)
                self.fixed_noise_t['noise'][batch_idx] = noise.cpu()
                self.fixed_noise_t['ts'][batch_idx] = ts.cpu()
        else:
            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            ts = th.randint(0, self.noise_scheduler.num_diff_steps, (batch_size,), device=self.device).long()
            noise = th.randn_like(x)

        x_noisy = self.noise_scheduler.q_sample(x_0=x, ts=ts, noise=noise)
        if self.require_g:
            x_noisy = x_noisy.requires_grad_(True)
            with th.inference_mode(False):
                predicted_noise = self.model(x_noisy, ts)
        else:
            predicted_noise = self.model(x_noisy, ts)
        loss = self.loss_f(noise, predicted_noise)
        # self.log("val_loss", loss)
        self.val_loss += loss.detach().cpu().item()
        self.i_batch_val += 1
        self.log("val_loss",self.val_loss / self.i_batch_val)
        return loss

    def on_validation_epoch_end(self):
        print(" {}. Validation Loss: {}".format(self.i_epoch, self.val_loss / self.i_batch_val))
        self.val_loss = 0.0
        self.i_batch_val = 0
