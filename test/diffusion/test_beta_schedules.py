"""Test diffusion parameter arithmetics"""
import unittest
import torch as th
from src.diffusion.beta_schedules import linear_beta_schedule, improved_beta_schedule, respaced_beta_schedule


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
        self.assertTrue(th.all(th.isclose(improved_beta_schedule(5), betas)))


class RespacedBetaSchedules(unittest.TestCase):
    def test_sparse_factor_one_equal(self):
        T = 1000
        schedules = (linear_beta_schedule, improved_beta_schedule)
        for sch in schedules:
            betas = sch(num_timesteps=T)
            respaced_betas, time_steps = respaced_beta_schedule(original_betas=betas, T=T, respaced_T=T)

            self.assertTrue(th.all(time_steps == th.arange(0, T)))
            self.assertTrue(th.allclose(betas, respaced_betas, atol=1e-2))
