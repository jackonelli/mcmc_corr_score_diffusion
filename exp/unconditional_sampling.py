"""Prototyping script for sampling a UNet-based unconditional diffusion model for MNIST"""


import sys


sys.path.append(".")
from pathlib import Path
from argparse import ArgumentParser
import torch as th
from src.diffusion.base import DiffusionSampler
from src.diffusion.beta_schedules import improved_beta_schedule, sparse_beta_schedule
from src.model.unet import load_mnist_diff
from src.model.imagenet import load_imagenet_diff
from src.utils.net import get_device, Device
from src.utils.vis import plot_samples_grid


def main():
    args = parse_args()
    device = get_device(Device.GPU)
    models_dir = Path.cwd() / "models"
    model_path = models_dir / f"{args.model}.pt"
    assert model_path.exists(), f"Model '{model_path}' does not exist."

    if "imagenet" in args.model:
        diff_model = load_imagenet_diff(model_path, device)
        channels, image_size = 3, 112
    elif "mnist" in args.model:
        diff_model = load_mnist_diff(model_path, device)
        channels, image_size = 1, 28
    else:
        raise ValueError("Incorrect model name '{args.model}'")
    T = args.num_diff_steps
    # beta_schedule = partial(_sparse_betas, og_schedule=improved_beta_schedule, og_num_diff_steps=1000)
    beta_schedule = improved_beta_schedule
    diff_sampler = DiffusionSampler(beta_schedule, T)

    samples, _ = diff_sampler.sample(
        diff_model, args.num_samples, device, (channels, image_size, image_size), verbose=True
    )
    if args.plot:
        plot_samples_grid(samples.detach().cpu().numpy())

    th.save(samples, Path.cwd() / "outputs" / f"uncond_samples_{args.model}.th")


def _sparse_betas(num_timesteps: int, og_schedule, og_num_diff_steps: int) -> th.Tensor:
    """Helper function that generate a beta schedule"""
    og_betas = og_schedule(og_num_diff_steps)
    return sparse_beta_schedule(og_betas, og_num_diff_steps // num_timesteps)


def parse_args():
    parser = ArgumentParser(prog="Train reconstruction classifier")
    parser.add_argument("--num_samples", default=100, type=int, help="Number of samples")
    parser.add_argument("--num_diff_steps", default=1000, type=int, help="Number of diffusion steps")
    parser.add_argument("--model", type=str, help="Model file (withouth '.pt' extension)")
    parser.add_argument("--plot", action="store_true", help="enables plots")
    return parser.parse_args()


if __name__ == "__main__":
    main()
