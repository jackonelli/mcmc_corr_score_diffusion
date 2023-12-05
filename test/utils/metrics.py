"""Test classification helpers"""
import unittest
import torch as th
from src.utils.metrics import mahalanobis, mahalanobis_diagonal


class Metrics(unittest.TestCase):
    def test_mahalanobis_diag_dist(self):
        u = th.randn((10,))
        v = th.randn_like(u)
        diag_covs = th.randn_like(u).exp()
        diag_dist = mahalanobis_diagonal(u, v, diag_covs)
        full_mat = th.diag(diag_covs.clone())
        full_dist = mahalanobis(u, v, full_mat)
        self.assertTrue(th.allclose(diag_dist, full_dist))

    def test_mahalanobis_diag_dist_batch(self):
        # u = th.randn((2, 10))
        # v = th.randn_like(u)
        # diag_covs = th.randn_like(u).exp()
        # diag_dist = mahalanobis_diagonal(u, v, diag_covs)
        # full_mat = th.diag(diag_covs.clone())
        # full_dist = mahalanobis(u, v, full_mat)
        # self.assertTrue(th.allclose(diag_dist, full_dist))
        pass
