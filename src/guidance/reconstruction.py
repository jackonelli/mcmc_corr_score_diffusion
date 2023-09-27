"""Reconstruction guidance"""
from collections.abc import Callable
import torch as th
from torch import nn
from src.diffusion.base import DiffusionSampler, extract
import pytorch_lightning as pl
from src.guidance.base import Guidance
from src.utils.classification import logits_to_log_prob


class ReconstructionGuidance(Guidance):
    """A class the computes the gradient used in reconstruction guidance"""

    def __init__(self, noise_pred: nn.Module, classifier: nn.Module, alpha_bars: th.Tensor, lambda_: float = 1.0):
        """
        @param noise_pred: eps_theta(x_t, t), noise prediction model
        @param classifier: Classifier model p(y|x_0)
        @param lambda_: Magnitude of the gradient
        """
        super(ReconstructionGuidance, self).__init__(lambda_=lambda_)
        self.classifier = classifier
        self.noise_pred = noise_pred
        self.alpha_bars = alpha_bars

    @th.no_grad()
    def grad(self, x_t, t, y, pred_noise, scale=False):
        """Compute score function for the classifier

        Estimates the score grad_x_t log p(y | x_t) by mapping x_t to x_0
        and then evaluating the given likelihood p(y | x_0)
        """
        if self.lambda_ > 0.0:
            th.set_grad_enabled(True)
            expectation = False

            if expectation:
                # I do not know if this is correct, or even necessary.
                x_t = x_t.clone().detach().requires_grad_(True)
                x_0 = self._map_to_x_0(x_t, t, pred_noise)
                logits = self.classifier(x_0)
                log_p = logits_to_log_prob(logits)
                # Get the log. probabilities of the correct classes
                y_log_probs = log_p[th.arange(log_p.size(0)), y]
                avg_log = y_log_probs.mean()
                grad_ = th.autograd.grad(avg_log, x_t, retain_graph=True)[0]
            else:
                grad_ = th.empty(x_t.shape, device=x_t.device)
                for i in range(grad_.shape[0]):
                    x_t_i = x_t[i : i + 1].clone().detach().requires_grad_(True)
                    x_0_i = self._map_to_x_0(x_t_i, t[i : i + 1], pred_noise[i : i + 1])
                    logits = self.classifier(x_0_i)
                    log_p = logits_to_log_prob(logits)
                    y_log_probs = log_p[th.arange(log_p.size(0)), y[i : i + 1]]
                    grad_[i] = th.autograd.grad(y_log_probs, x_t_i, retain_graph=True)[0]

            s = 1.0
            if scale:
                s = th.norm(pred_noise) / grad_.norm()
            grad_ = self.lambda_ * grad_ * s
            th.set_grad_enabled(False)
        else:
            grad_ = th.zeros_like(x_t)
        return grad_

    def predict_x_0(self, x_t, t):
        t_tensor = th.full((x_t.shape[0],), t, device=x_t.device)
        pred_noise_t = self.noise_pred(x_t, t_tensor)
        return self._map_to_x_0(x_t, t_tensor, pred_noise_t)

    def _map_to_x_0(self, x_t, t, pred_noise_t):
        """Map x_t to x_0

        For reconstruction guidance, we assume that we only have access to the likelihood
        p(y | x_0); we approximate p(y | x_t) ~= p(y | x_0_hat(x_t))
        """
        return mean_x_0_given_x_t(x_t, pred_noise_t, self.alpha_bars[t])


def mean_x_0_given_x_t(x_t: th.Tensor, noise_pred_t: th.Tensor, a_bar_t: th.Tensor):
    """Compute E[x_0|x_t] using Tweedie's formula

    See Prop. 1 in https://arxiv.org/pdf/2209.14687.pdf

    NB: This uses the noise prediction function eps_theta, not the score function s_theta.
    """
    # TODO: Fix this
    assert all(a_bar_t == a_bar_t[0])
    _a = a_bar_t[0]
    return (x_t - th.sqrt(1.0 - _a) * noise_pred_t) / th.sqrt(_a)


class ReconstructionClassifier(pl.LightningModule):
    """Train classifier for reconstruction guidance."""

    def __init__(self, model: nn.Module, loss_f: Callable):
        super().__init__()
        self.model = model
        self.loss_f = loss_f

        # Default Initialization
        self.train_loss = 0.0
        self.val_loss = 0.0
        self.i_batch_train = 0
        self.i_batch_val = 0
        self.i_epoch = 0

    def training_step(self, batch, _):
        batch_size = batch["pixel_values"].shape[0]
        x = batch["pixel_values"].to(self.device).float()
        y = batch["label"].to(self.device).long()
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

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = th.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):
        batch_size = batch["pixel_values"].shape[0]
        x = batch["pixel_values"].to(self.device)
        y = batch["label"].to(self.device)

        # Unsure why/if this is needed.
        rng_state = th.get_rng_state()
        th.manual_seed(self.i_batch_val)
        th.set_rng_state(rng_state)

        predicted_y = self.model(x)

        loss = self.loss_f(predicted_y, y)
        self.log("val_loss", loss)
        self.val_loss += loss.detach().cpu().item()
        self.i_batch_val += 1
        return loss

    def on_validation_epoch_end(self):
        print(" {}. Validation Loss: {}".format(self.i_epoch, self.val_loss / self.i_batch_val))
        self.val_loss = 0.0
        self.i_batch_val = 0
