"""Prototyping script for sampling a UNet-based unconditional diffusion model for MNIST"""


from pathlib import Path
import torch as th
from src.diffusion.base import DiffusionSampler
from src.diffusion.beta_schedules import improved_beta_schedule
from src.model.unet import UNet, load_mnist_diff
from src.utils.net import get_device, Device
from src.utils.vis import plot_samples


def main():
    image_size = 28
    time_emb_dim = 112
    channels = 1
    num_diff_steps = 1000

    dev = get_device(Device.GPU)
    models_dir = Path.cwd() / "models"
    unet = load_mnist_diff(models_dir / "uncond_unet_mnist.pt", dev)
    diff_sampler = DiffusionSampler(improved_beta_schedule, num_diff_steps)

    samples, _ = diff_sampler.sample(unet, 100, dev, (1, 28, 28))
    plot_samples(samples)


if __name__ == "__main__":
    main()
