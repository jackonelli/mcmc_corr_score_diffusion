"""Guidance"""
import torch as th
from torch import nn
from abc import ABC, abstractmethod


class Guidance(ABC):
    def __init__(self, lambda_):
        self.lambda_ = lambda_

    @abstractmethod
    def grad(self, *args, **kwargs):
        raise NotImplementedError


class ReconstructionGuidance(Guidance):
    """A class the computes the gradient used in reconstruction guidance"""

    def __init__(
        self, classifier: nn.Module, noise_pred: nn.Module, alpha_bars: th.Tensor, loss: nn.Module, lambda_: float = 1.0
    ):
        """
        @param classifier: Classifier model p(y|x_0)
        @param noise_pred: eps_theta(x_t, t), noise prediction model
        @param loss: Corresponding loss function for the classifier (e.g., CrossEntropy)
        @param lambda_: Magnitude of the gradient
        """
        super(ReconstructionGuidance, self).__init__(lambda_=lambda_)
        self.classifier = classifier
        self.noise_pred = noise_pred
        self.loss = loss
        self.alpha_bars = alpha_bars

    def grad(self, x_t, t, y):
        """Compute score function for the classifier

        Estimates the score grad_x_t log p(y | x_t) by mapping x_t to x_0
        and then evaluating the given likelihood p(y | x_0)
        """
        x_t.requires_grad = True
        x_0 = self._map_to_x_0(x_t, t)recon
        loss = self.loss(self.classifier(x_0), y)
        return self.lambda_ * th.autograd.grad(loss, x_t, retain_graph=True)[0]

    def _map_to_x_0(self, x_t, t):
        """Map x_t to x_0

        For reconstruction guidance, we assume that we only have access to the likelihood
        p(y | x_0); we approximate p(y | x_t) ~= p(y | x_0_hat(x_t))
        """
        return mean_x_0_given_x_t(x_t, self.noise_pred(x_t, t), self.alpha_bars[t])


def mean_x_0_given_x_t(x_t, noise_pred_t, a_bar_t):
    """Compute E[x_0|x_t] using Tweedie's formula

    See Prop. 1 in https://arxiv.org/pdf/2209.14687.pdf

    NB: This uses the score function eps_theta
    """
    return (x_t - th.sqrt(1.0 - a_bar_t) * noise_pred_t) / th.sqrt(a_bar_t)


class ClassifierFullGuidance(Guidance):
    def __init__(self, classifier: nn.Module, loss: nn.Module, lambda_: float = 1.0):
        """
        @param classifier: Classifier model p(y|x_t , t)
        @param loss: Corresponding loss for the classifier
        @param lambda_: Magnitude of the gradient
        """
        super(ClassifierFullGuidance, self).__init__(lambda_=lambda_)
        self.classifier = classifier
        self.loss = loss

    def grad(self, x_t, t, y):
        x_t.requires_grad = True
        loss = self.loss(self.classifier(x_t, t), y)
        return self.lambda_ * th.autograd.grad(loss, x_t, retain_graph=True)[0]
