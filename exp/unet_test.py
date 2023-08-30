"""Test script for UNet

Create instance and feed a random tensor of MNIST shape to it.
"""
from pathlib import Path
import torch as th
from src.model.unet import UNetModel, attention_down_sampling
from src.utils.net import dev
from src.samplers.sampling import reverse_diffusion


def main():
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
    model.load_state_dict(
        path=Path.cwd() / "models/model060000.pt",
    )
    device = dev()
    model.to(device)
    reverse_diffusion(model, image_size, alpha_ts, sigma_ts, True)


if __name__ == "__main__":
    main()
