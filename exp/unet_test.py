"""Test script for UNet

Create instance and feed a random tensor of MNIST shape to it.
"""
from pathlib import Path
import torch as th
from src.diffusion.beta_schedules import improved_beta_schedule, linear_beta_schedule
from src.model.unet import UNetModel, attention_down_sampling
from src.utils.net import dev, Device
from src.samplers.sampling import reverse_diffusion
import matplotlib.pyplot as plt

EXP_NAME = "uncond_rev_diff"


def main():
    T = 100
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
    device = dev(Device.GPU)
    model.to(device)
    print(f"Using device {device}")

    bs = linear_beta_schedule(num_timesteps=T).to(device)
    as_ = 1.0 - bs
    ss = bs
    x_0, _ = reverse_diffusion(model, image_size, as_, ss, True)
    x_0 = x_0.detach().cpu()
    save_dir = Path.cwd() / "outputs" / EXP_NAME
    save_dir.mkdir(parents=True, exist_ok=True)
    th.save(x_0, save_dir / f"x_0_T{T}.pth")
    plt.imshow(x_0[0, 0, :, :])
    plt.show()


if __name__ == "__main__":
    main()
