"""Test TDS UNet instantiation

Create instance and feed a random tensor of MNIST shape to it.
"""
import unittest
import torch as th
from src.model.tds_unet import UNetModel, attention_down_sampling
from src.utils.net import dev, Device


class UNetInstantiation(unittest.TestCase):
    def test_init(self):
        T = 10
        image_channels = 1
        image_size = 28
        model = UNetModel(
            image_size=image_size,
            in_channels=image_channels,
            model_channels=64,
            out_channels=2 * image_channels,  # Times 2 if learned sigma
            num_res_blocks=3,
            attention_resolutions=attention_down_sampling((28, 14, 7), image_size),
            dropout=0,
            channel_mult=(1, 2, 2, 2),
            num_classes=None,
            use_checkpoint=False,
            num_heads=4,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_fp16=False,
            use_new_attention_order=False,
            diffusion_steps=T,
        )
        device = dev(Device.CPU)
        model.to(device)

        batch_size = 2
        x = th.rand((batch_size, image_channels, image_size, image_size))
        ts = th.randint(low=1, high=T, size=(batch_size,))
        x, ts = x.to(device), ts.to(device)
        self.assertEqual(x.device, th.device("cpu"))

        out = model(x, ts)
        self.assertEqual(out.size(), th.Size((2, 2 * image_channels, image_size, image_size)))
