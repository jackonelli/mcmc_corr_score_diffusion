"""Test diffusion parameter arithmetics"""
import unittest
import torch as th
from src.diffusion.beta_schedules import linear_beta_schedule, beta_schedule_improved


class BetaShedules(unittest.TestCase):
    def test_linear_beta_schedule(self):
        beta_start = 0.001
        beta_end = 0.02
        num_steps = 5

        betas_true = th.tensor([0.0010, 0.0058, 0.0105, 0.0152, 0.0200])
        betas_schedule = linear_beta_schedule(beta_start, beta_end, num_steps)

        self.assertTrue(th.allclose(betas_schedule, betas_true, atol=1e-2))

    def test_improved_beta_schedule(self):
        betas = th.tensor([0.1012940794, 0.2795438460, 0.4736353534, 0.7240523691, 0.9990000000])
        self.assertTrue(th.all(th.isclose(beta_schedule_improved(5), betas)))
