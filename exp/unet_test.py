"""Prototyping script for sampling a UNet-based unconditional diffusion model for MNIST"""


import sys

sys.path.append(".")
from pathlib import Path
from argparse import ArgumentParser
from src.diffusion.base import DiffusionSampler
from src.diffusion.beta_schedules import improved_beta_schedule
from src.model.unet import load_mnist_diff
from src.utils.net import get_device, Device
from src.utils.vis import plot_samples_grid


def main():
    num_diff_steps = 1000
    args = parse_args()
    device = get_device(Device.GPU)
    models_dir = Path.cwd() / "models"
    unet = load_mnist_diff(models_dir / "uncond_unet_mnist.pt", device)
    diff_sampler = DiffusionSampler(improved_beta_schedule, num_diff_steps)

    samples, _ = diff_sampler.sample(unet, 100, device, (1, 28, 28))
    if args.plot:
        plot_samples_grid(samples)


def parse_args():
    parser = ArgumentParser(prog="Train reconstruction classifier")
    parser.add_argument("--num_samples", default=100, type=int, help="Number of samples")
    parser.add_argument("--plot", action="store_true", help="enables plots")
    return parser.parse_args()


if __name__ == "__main__":
    main()
