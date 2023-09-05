"""Test reconstruction guidance"""
import unittest
from pathlib import Path
import torch as th
import torch.nn.functional as F
from src.model.resnet import load_classifier
from src.utils.net import get_device, Device


class TestReconstructionGuidance(unittest.TestCase):
    def test_loss_fn(self):
        num_samples, num_classes = 5, 10
        loss = F.cross_entropy
        classes = th.ones((num_samples,), dtype=th.int64)
        logits = th.randn((num_samples, num_classes))
        loss(logits, classes)

    def test_grad(self):
        batch_size, num_classes = 2, 10
        # x = th.randn((batch_size, 1, 28, 28), requires_grad=True)
        logits = th.randn((batch_size, num_classes), requires_grad=True)
        y = th.ones((batch_size,), dtype=th.int64)
        loss = F.cross_entropy
        l = loss(logits, y)
        th.autograd.grad(l, logits, retain_graph=True)

    def test_init(self):
        image_channels = 1
        image_size = 28
        model_path = Path.cwd() / "models/resnet_reconstruction_classifier_mnist.pt"
        model = load_classifier(model_path)
        device = get_device(Device.GPU)
        model.to(device)

        batch_size = 2
        x = th.rand((batch_size, image_channels, image_size, image_size))
        x = x.to(device)
        out = model(x)
        self.assertEqual(out.size(), th.Size((2, 10)))
