"""Test classification helpers"""
import unittest
import torch as th
from src.utils.classification import logits_to_log_prob, logits_to_log_prob_mean


class Classification(unittest.TestCase):
    def test_logits_to_log_prob(self):
        logits = th.randn((100, 10))
        p = logits_to_log_prob(logits)
        self.assertEqual(logits.size(), p.size())
        self.assertFalse(th.any(th.exp(p) > 1.0))
        self.assertFalse(th.any(th.exp(p) < 0.0))
        self.assertTrue(th.allclose(th.exp(p).sum(dim=1), th.ones((100,))))

    def test_logits_to_log_prob_mean(self):
        n_samples = 100
        dim = 10
        n_ensembles = 5
        logits = th.randn((n_samples, dim, n_ensembles))
        p = logits_to_log_prob_mean(logits)
        self.assertEqual(logits.size()[:2], p.size())
        self.assertFalse(th.any(th.exp(p) > 1.0))
        self.assertFalse(th.any(th.exp(p) < 0.0))
        self.assertTrue(th.allclose(th.exp(p).sum(dim=1), th.ones((100,))))
