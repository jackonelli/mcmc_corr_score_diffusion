"""Test ResNet instantiation

Create instance and feed a random tensor of MNIST shape to it.
"""
import unittest
from pathlib import Path
import torch as th
import torch.nn as nn
from src.model.resnet import ResNet, Bottleneck, load_classifier
from src.utils.net import get_device, Device


class ResNetInstantiation(unittest.TestCase):
    def test_init(self):
        image_channels = 1
        image_size = 28
        model_path = Path.cwd() / "models/resnet.pth.tar"
        model = load_classifier(model_path)
        # NB: This fails for device CPU
        # There is something about the DataParallel super class which prevents CPU.
        device = get_device(Device.GPU)
        model.to(device)

        batch_size = 2
        x = th.rand((batch_size, image_channels, image_size, image_size))
        x = x.to(device)
        out = model(x)
        self.assertEqual(out.size(), th.Size((2, 10)))
