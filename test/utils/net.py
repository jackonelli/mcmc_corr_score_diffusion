"""Test NN helpers"""
import unittest
import torch as th

from src.utils.net import batch_grad
from src.utils.classification import logits_to_log_prob


class Nn(unittest.TestCase):
    def test_batch_grad(self):
        # Test aggregating operation
        x = th.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        f = x.sum(dim=1)
        grads = batch_grad(f, x)
        self.assertTrue(th.allclose(grads, th.ones((2, 3))))

        # Test non-aggregating operation
        x = th.randn((4, 1), requires_grad=True)
        f = x.exp()
        grads = batch_grad(f, x)
        self.assertTrue(th.allclose(grads, x.exp()))

        # Compare with Anders' loop version
        y = th.ones((5,), dtype=th.int64)
        th.set_grad_enabled(True)
        # I do not know if this is correct, or even necessary.
        src = th.randn((5, 7), requires_grad=True)
        x_t = src.clone().requires_grad_(True)
        logits = x_t
        log_p = logits_to_log_prob(logits)
        # Get the log. probabilities of the correct classes
        y_log_probs = log_p[th.arange(log_p.size(0)), y]
        grad_vectorised = batch_grad(y_log_probs, x_t)

        x_t = src.clone().requires_grad_(True)
        grad_loop = th.empty(x_t.shape, device=x_t.device)
        for i in range(grad_loop.shape[0]):
            x_t_i = x_t[i : i + 1].clone().detach().requires_grad_(True)
            logits = x_t_i
            log_p = logits_to_log_prob(logits)
            y_log_probs = log_p[th.arange(log_p.size(0)), y[i : i + 1]]
            grad_loop[i] = th.autograd.grad(y_log_probs, x_t_i, retain_graph=True)[0]
        self.assertTrue(th.allclose(grad_vectorised, grad_loop))
