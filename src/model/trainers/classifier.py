from typing import Tuple
from collections.abc import Callable
import torch as th
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from src.utils.metrics import accuracy, hard_label_from_logit


def _process_labelled_batch(batch, device) -> Tuple[int, th.Tensor, th.Tensor]:
    """Hack to handle multiple formats of dataloaders"""
    if isinstance(batch, dict):
        batch_size = batch["x"].size(0)
        x = batch["x"].to(device)
        y = batch["labels"].long().to(device)
    else:
        # Discard label
        x, y = batch
        x = x.to(device)
        y = y.long().to(device)
        batch_size = x.size(0)
    return batch_size, x, y


def process_labelled_batch_cifar100(batch, device) -> Tuple[int, th.Tensor, th.Tensor]:
    """Hack to handle Cifar100 data"""
    batch_size = batch["pixel_values"].size(0)
    x = batch["pixel_values"].to(device)
    y = batch["fine_label"].long().to(device)
    return batch_size, x, y

def process_labelled_batch_cifar10(batch, device) -> Tuple[int, th.Tensor, th.Tensor]:
    """Hack to handle Cifar10 data"""
    batch_size = batch["pixel_values"].size(0)
    x = batch["pixel_values"].to(device)
    y = batch["label"].long().to(device)
    return batch_size, x, y


class DiffusionClassifier(pl.LightningModule):
    """Trainer for learning classification model p(y | x_t),
    where x_t is a noisy sample from a forward diffusion process."""

    def __init__(
        self, model: nn.Module, loss_f: Callable, noise_scheduler, batches_per_epoch, batch_fn=_process_labelled_batch,
            lr=2e-4, weight_decay=0.
    ):
        super().__init__()
        self.model = model
        self.loss_f = loss_f
        self.noise_scheduler = noise_scheduler

        # Optimizer
        self.lr = lr
        self.weight_decay = weight_decay

        # Default Initialization
        self.train_loss = 0.0
        self.val_loss = 0.0
        self.val_loss0 = 0.0
        self.val_acc0 = 0.0
        self.i_batch_train = 0
        self.i_batch_val = 0
        self.i_epoch = 0
        self._batch_fn = batch_fn
        self._batches_per_epoch = batches_per_epoch

    def training_step(self, batch, batch_idx):
        batch_size, x, y = self._batch_fn(batch, self.device)
        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        T = self.noise_scheduler.time_steps.size(0)
        ts = th.randint(0, T, (batch_size,), device=self.device).long()
        # ts = th.zeros((batch_size,), device=self.device).long()
        # p = np.linspace(5, 1, T)
        # p = p/p.sum()
        # ts = np.random.choice(T, (batch_size,), p=p)
        # ts = th.from_numpy(ts).long().to(self.device)

        noise = th.randn_like(x)
        x_noisy = self.noise_scheduler.q_sample(x_0=x, ts=ts, noise=noise)
        predicted_y = self.model(x_noisy, ts)
        loss = self.loss_f(predicted_y, y)
        self.train_loss += loss.detach().cpu().item()
        self.i_batch_train += 1
        self.log("train_loss", self.train_loss / self.i_batch_train)
        return loss

    def on_train_epoch_end(self):
        print(" {}. Train Loss: {}".format(self.i_epoch, self.train_loss / self.i_batch_train))
        self.train_loss = 0.0
        self.i_batch_train = 0
        self.i_epoch += 1

    def validation_step(self, batch, batch_idx):
        batch_size, x, y = self._batch_fn(batch, self.device)

        # rng_state = th.get_rng_state()
        # th.manual_seed(self.i_batch_val)

        T = self.noise_scheduler.time_steps.size(0)
        # Only report val. acc for t=0
        ts = th.randint(0, T, (batch_size,), device=self.device).long()
        ts0 = th.zeros((batch_size,), device=self.device).long()
        # th.set_rng_state(rng_state)
        logits0 = self.model(x, ts0)
        loss0 = self.loss_f(logits0, y)
        acc0 = accuracy(hard_label_from_logit(logits0), y)

        logits = self.model(x, ts)
        loss = self.loss_f(logits, y)

        self.val_loss += loss.detach().cpu().item()
        self.val_loss0 += loss0.detach().cpu().item()
        self.val_acc0 += acc0.detach().cpu().item()
        self.i_batch_val += 1

        val_loss0 = self.val_loss0 / self.i_batch_val
        val_loss = self.val_loss / self.i_batch_val
        val_acc_pct_0 = (self.val_acc0 / self.i_batch_val) * 100
        self.log("val_loss", val_loss)
        self.log("val_loss_0", val_loss0)
        self.log("acc_0", val_acc_pct_0)
        return loss0

    def on_validation_epoch_end(self):
        val_loss0 = self.val_loss0 / self.i_batch_val
        val_loss = self.val_loss / self.i_batch_val
        val_acc_pct_0 = (self.val_acc0 / self.i_batch_val) * 100
        print(f" {self.i_epoch}. Val. Loss at t=0: {val_loss0:.2f}, Val. acc at t=0: {val_acc_pct_0:.1f}%, Val loss: {val_loss:.2f}")
        self.val_loss = 0.0
        self.val_loss0 = 0.0
        self.val_acc0 = 0.0
        self.i_batch_val = 0

    def configure_optimizers(self):
        warmup = 5000
        def warmup_lr(step):
            return min(step, warmup) / warmup
        optimizer = th.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # scheduler = th.optim.lr_scheduler.StepLR(optimizer, 1, gamma=1.0)
        scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
        }}


class StandardClassifier(pl.LightningModule):
    """Trainer for learning a standard classification model p(y | x),

    i.e., with no dependence on time t.
    This is merely for debugging purposes.
    """

    def __init__(self, model: nn.Module, loss_f: Callable, batches_per_epoch, batch_fn=_process_labelled_batch):
        super().__init__()
        self.model = model
        self.loss_f = loss_f

        # Default Initialization
        self.train_loss = 0.0
        self.val_loss = 0.0
        self.val_acc = 0.0
        self.i_batch_train = 0
        self.i_batch_val = 0
        self.i_epoch = 0
        self._batch_fn = batch_fn
        self._batches_per_epoch = batches_per_epoch

        self.weight_decay = 0.
        self.lr = 2e-4

    def training_step(self, batch, batch_idx):
        batch_size, x, y = self._batch_fn(batch, self.device)
        predicted_y = self.model(x)
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

    def validation_step(self, batch, batch_idx):
        batch_size, x, y = self._batch_fn(batch, self.device)

        logits = self.model(x)
        loss = self.loss_f(logits, y)
        acc = accuracy(hard_label_from_logit(logits), y)

        self.log("val_loss", loss)
        self.log("acc", acc)
        self.val_loss += loss.detach().cpu().item()
        self.val_acc += acc.detach().cpu().item()
        self.i_batch_val += 1
        return loss

    def on_validation_epoch_end(self):
        print(
            f" {self.i_epoch}. Val. Loss: {self.val_loss / self.i_batch_val}, Val. acc at t=0: {(self.val_acc / self.i_batch_val)*100:.1f}%"
        )
        self.val_loss = 0.0
        self.val_acc = 0.0
        self.i_batch_val = 0

    def configure_optimizers(self):
        warmup = 5000
        def warmup_lr(step):
            return min(step, warmup) / warmup

        optimizer = th.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }}
