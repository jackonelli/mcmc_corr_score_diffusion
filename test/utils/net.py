"""Test NN helpers"""
import unittest
import torch as th

from src.utils.net import batch_grad


class Nn(unittest.TestCase):
    def test_batch_grad(self):
        x = th.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        f = x.sum(dim=1)
        grads = batch_grad(f, x)
        self.assertTrue(th.allclose(grads, th.ones((2, 3))))

        x = th.randn((4, 1), requires_grad=True)
        f = x.exp()
        grads = batch_grad(f, x)
        self.assertTrue(th.allclose(grads, x.exp()))
