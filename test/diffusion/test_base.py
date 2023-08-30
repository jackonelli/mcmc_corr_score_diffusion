"""Test diffusion parameter arithmetics"""
import unittest
import torch as th
from src.diffusion.base import sample_x_t_given_x_0


class DiffusionParameters(unittest.TestCase):
    def test_produce_noisy_samples(self):
        ts = (
            th.Tensor([2, 3, 2, 1])
            .long()
            .reshape(
                -1,
            )
        )
        x_0 = th.Tensor([3, 3, 3, 3]).reshape(-1, 1)
        betas = th.Tensor([0.1, 0.2, 0.3, 0.4]).reshape(
            -1,
        )
        alphas = 1 - betas
        alphas_bar = th.cumprod(alphas, dim=0)

        noisy_test = sample_x_t_given_x_0(x_0, ts, alphas_bar)

        self.assertEqual(noisy_test.size(), x_0.size())
