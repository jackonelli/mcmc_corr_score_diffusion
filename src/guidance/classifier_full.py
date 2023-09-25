"""Classifier-full guidance

'Classifier-full' is a term we use for the type of guidance where we assume access to an (approx.) likelihood function
p(y | x_t), forall t = 0, ..., T.

That is, we need access to a classifier for all noise levels.
"""
import torch as th
from torch import nn
from src.guidance.base import Guidance
from src.utils.classification import logits_to_log_prob


class ClassifierFullGuidance(Guidance):
    def __init__(self, classifier: nn.Module, lambda_: float = 1.0):
        """
        @param classifier: Classifier model p(y|x_t , t)
        @param lambda_: Magnitude of the gradient
        """
        super(ClassifierFullGuidance, self).__init__(lambda_=lambda_)
        self.classifier = classifier

    @th.no_grad()
    def grad(self, x_t, t, y, pred_noise, scale=False):
        """Compute score function for the classifier

        """
        if self.lambda_ > 0.0:
            th.set_grad_enabled(True)
            expectation = False

            if expectation:
                # I do not know if this is correct, or even necessary.
                x_t = x_t.clone().detach().requires_grad_(True)
                logits = self.classifier(x_t, t)
                log_p = logits_to_log_prob(logits)
                # Get the log. probabilities of the correct classes
                y_log_probs = log_p[th.arange(log_p.size(0)), y]
                avg_log = y_log_probs.mean()
                grad_ = th.autograd.grad(avg_log, x_t, retain_graph=True)[0]
            else:
                grad_ = th.empty(x_t.shape, device=x_t.device)
                for i in range(grad_.shape[0]):
                    x_t_i = x_t[i:i+1].clone().detach().requires_grad_(True)
                    logits = self.classifier(x_t_i, t[i:i+1])
                    log_p = logits_to_log_prob(logits)
                    y_log_probs = log_p[th.arange(log_p.size(0)), y[i:i+1]]
                    grad_[i] = th.autograd.grad(y_log_probs, x_t_i, retain_graph=True)[0]

            s = 1.
            if scale:
                s = th.norm(pred_noise) / grad_.norm()
            grad_ = self.lambda_ * grad_ * s
            th.set_grad_enabled(False)
        else:
            grad_ = th.zeros_like(x_t)
        return grad_
