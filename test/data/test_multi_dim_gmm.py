"""Test multi-dimensional GMM data for composition"""
import unittest
import torch as th
from torch.distributions.multivariate_normal import MultivariateNormal
from src.data.multi_dim_gmm import Gmm


class TestGmm(unittest.TestCase):
    def test_isotropic_nll_single_mixt_comp(self):
        means = th.ones(1, 100)
        num_comp, x_dim = means.size()
        assert num_comp == 1
        var = num_comp * [0.09 * th.eye(x_dim)]
        data = Gmm(means, var)
        sample, _ = data.sample(100)
        assert sample.size() == (100, x_dim)

        ref = MultivariateNormal(loc=data.means[0], covariance_matrix=data.covs[0])
        self.assertAlmostEqual(-ref.log_prob(sample).mean().item(), data.isotropic_nll(sample), places=4)

    def test_full_nll_single_mixt_comp(self):
        means = th.ones(1, 100)
        num_comp, x_dim = means.size()
        assert num_comp == 1
        var = num_comp * [0.09 * th.eye(x_dim)]
        data = Gmm(means, var)
        sample, _ = data.sample(100)
        assert sample.size() == (100, x_dim)

        ref = MultivariateNormal(loc=data.means[0], covariance_matrix=data.covs[0])
        self.assertAlmostEqual(-ref.log_prob(sample).mean().item(), data.full_nll(sample), places=4)

    def test_compare_full_isotropic_nll(self):
        means = th.ones(10, 8)
        num_comp, x_dim = means.size()
        var = num_comp * [0.09 * th.eye(x_dim)]
        data = Gmm(means, var)
        sample, _ = data.sample(100)
        assert sample.size() == (100, x_dim)

        self.assertAlmostEqual(data.isotropic_nll(sample), data.full_nll(sample), places=4)
