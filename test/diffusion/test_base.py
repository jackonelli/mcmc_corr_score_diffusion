"""Test diffusion parameter arithmetics"""
import unittest
import torch as th
from src.diffusion.base import linear_beta_schedule, sample_x_t_given_x_0


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

    def test_linear_beta_schedule(self):
        beta_start = 0.001
        beta_end = 0.02
        num_steps = 5

        betas_true = th.tensor([0.0010, 0.0058, 0.0105, 0.0152, 0.0200])
        betas_schedule = linear_beta_schedule(beta_start, beta_end, num_steps)

        self.assertTrue(th.allclose(betas_schedule, betas_true, atol=1e-2))
