"""Prototyping script for sampling a UNet-based unconditional diffusion model for MNIST"""


import sys


sys.path.append(".")
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from src.diffusion.base import DiffusionSampler
from src.diffusion.beta_schedules import improved_beta_schedule
from src.model.imagenet import load_unet_from_state_dict, load_unet_from_checkpoint
from src.utils.net import get_device, Device
from src.data.utils import reverse_transform
import torch as th


def main():
    args = parse_args()
    device = get_device(Device.GPU)
    chkpt_dir = Path.cwd() / "lightning_logs/version_19/checkpoints/"
    chkpt = next(chkpt_dir.iterdir())
    assert chkpt.exists(), f"Model '{chkpt}' does not exist."

    diff_model = load_unet_from_checkpoint(chkpt, device)
    channels, image_size = 3, 112
    T = args.num_diff_steps
    # beta_schedule = partial(_sparse_betas, og_schedule=improved_beta_schedule, og_num_diff_steps=1000)
    betas = improved_beta_schedule(num_timesteps=T)
    time_steps = th.tensor([i for i in range(T)])
    diff_sampler = DiffusionSampler(betas, time_steps)

    samples, _ = diff_sampler.sample(
        diff_model, args.num_samples, device, (channels, image_size, image_size), verbose=True
    )
    print("Samples", samples.size())
    sampled_img = samples[0].detach().cpu()
    plt.imshow(reverse_transform(sampled_img))
    plt.show()
    img = reverse_transform(sampled_img)
    img.save("single_sample.jpg")


def parse_args():
    parser = ArgumentParser(prog="Sample from unconditional diff. model")
    parser.add_argument("--num_samples", default=1, type=int, help="Number of samples")
    parser.add_argument("--num_diff_steps", default=1000, type=int, help="Number of diffusion steps")
    return parser.parse_args()


if __name__ == "__main__":
    main()
