"""Test classification helpers"""
import unittest
import torch as th
from src.utils.metrics import mahalanobis, mahalanobis_diagonal, r3_accuracy


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

    def test_r3_accuracy(self):
        prob_vecs = th.tensor(
            [
                [0.6, 0.2, 0.2],
                [0.4, 0.3, 0.3],
                [0.2, 0.7, 0.1],
                [0.7, 0.2, 0.1],
            ]
        )
        true_classes = th.tensor([0, 0, 1, 1])
        self.assertAlmostEqual(0.5, r3_accuracy(prob_vecs, true_classes).item())
