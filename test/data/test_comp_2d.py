"""Test 2D data for composition"""
import unittest
import torch as th
from src.data.comp_2d import Gmm


class TestGmm2d(unittest.TestCase):
    def test_nll_single_mixt_comp(self):
        data = Gmm(1)
        sample, _ = data.sample(100)
        from torch.distributions.multivariate_normal import MultivariateNormal

        ref = MultivariateNormal(loc=data.means[0], covariance_matrix=data.std**2 * th.eye(2))
        self.assertAlmostEqual(-ref.log_prob(sample).mean().item(), data.nll(sample))
