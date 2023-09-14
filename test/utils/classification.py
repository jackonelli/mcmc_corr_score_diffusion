"""Test classification helpers"""
import unittest
import torch as th
from src.utils.classification import logits_to_prob_vec


class Classification(unittest.TestCase):
    def test_logits_to_prob_vec(self):
        logits = th.randn((100, 10))
        p = logits_to_prob_vec(logits)
        self.assertEqual(logits.size(), p.size())
        self.assertFalse(th.any(p > 1.0))
        self.assertFalse(th.any(p < 0.0))
        self.assertTrue(th.allclose(p.sum(dim=1), th.ones((100,))))
