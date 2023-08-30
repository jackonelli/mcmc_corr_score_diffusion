"""Test ResNet instantiation

Create instance and feed a random tensor of MNIST shape to it.
"""
import unittest
from pathlib import Path
import torch as th
import torch.nn as nn
from src.model.resnet import ResNet, Bottleneck
from src.utils.net import dev, Device


class ResNetInstantiation(unittest.TestCase):
    def test_init(self):
        T = 10
        image_channels = 1
        image_size = 28
        model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=10, num_channels=1)
        model = nn.DataParallel(model)
        resnet_model_info = th.load(Path.cwd() / "models/resnet.pth.tar", map_location="cpu")
        model.load_state_dict(resnet_model_info["state_dict"])
        # NB: This fails for device CPU
        # There is something about the DataParallel super class which prevents CPU.
        device = dev(Device.GPU)
        model.to(device)

        batch_size = 2
        x = th.rand((batch_size, image_channels, image_size, image_size))
        x = x.to(device)
        self.assertEqual(x.device, th.device("cpu"))

        out = model(x)
        self.assertEqual(out.size(), th.Size((2, 10)))
