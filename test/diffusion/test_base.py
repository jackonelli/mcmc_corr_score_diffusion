"""Test diffusion parameter arithmetics"""
import unittest
import torch as th
from src.diffusion.base import compute_alpha_bars, sample_x_t_given_x_0


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
        alphas_bar = compute_alpha_bars(alphas)
        noise = th.tensor([0.4199, 0.9844, 1.1147, 0.1688]).reshape(-1, 1)
        noisy_actual = th.tensor([2.4255, 2.4719, 2.9148, 2.6349]).reshape(-1, 1)

        noisy_test = sample_x_t_given_x_0(x_0, ts, alphas_bar, noise)

        self.assertEqual(noisy_test.size(), x_0.size())
        self.assertTrue(th.allclose(noisy_test, noisy_actual, atol=1e-3))
