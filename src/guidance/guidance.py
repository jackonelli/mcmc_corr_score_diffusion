import torch
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

    def __init__(self, classifier: nn.Module, diffusion_model: nn.Module, loss: nn.Module, lambda_: float = 1.0):
        """
        @param classifier: Classifier model p(y|x_0)
        @param diffusion_model: Diffusion model
        @param loss: Corresponding loss function for the classifier (e.g., CrossEntropy)
        @param lambda_: Magnitude of the gradient
        """
        super(ReconstructionGuidance, self).__init__(lambda_=lambda_)
        self.classifier = classifier
        self.diffusion_model = diffusion_model
        self.loss = loss
        self.alpha_bar = self.diffusion_model.alpha_bar

    def grad(self, x_t, t, y):
        x_t.requires_grad = True
        # Compute E[x_0|x_t]
        x_0 = (x_t - torch.sqrt(1.0 - self.alpha_bar[t]) * self.diffusion_model(x_t, t)) / torch.sqrt(self.alpha_bar[t])
        loss = self.loss(self.classifier(x_0), y)
        return self.lambda_ * torch.autograd.grad(loss, x_t, retain_graph=True)[0]


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
        return self.lambda_ * torch.autograd.grad(loss, x_t, retain_graph=True)[0]
