"""Test UNet instantiation

Create instance and feed a random tensor of MNIST shape to it.
"""
import unittest
import torch as th
from src.model.unet import UNet
from src.utils.net import dev, Device


class UNetInstantiation(unittest.TestCase):
    def test_init(self):
        T = 10
        image_channels = 1
        image_size = 28
        time_emb_dim = 112
        model = UNet(dim=image_size, time_dim=time_emb_dim, channels=image_channels)
        device = dev(Device.CPU)
        model.to(device)

        batch_size = 2
        x = th.rand((batch_size, image_channels, image_size, image_size))
        ts = th.randint(low=1, high=T, size=(batch_size,))
        x, ts = x.to(device), ts.to(device)
        self.assertEqual(x.device, th.device("cpu"))

        out = model(x, ts)
        self.assertEqual(out.size(), th.Size((2, image_channels, image_size, image_size)))
