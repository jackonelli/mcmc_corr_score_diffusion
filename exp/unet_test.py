"""Prototyping script for sampling a UNet-based unconditional diffusion model for MNIST"""


from pathlib import Path
import torch as th
import matplotlib.pyplot as plt
from src.diffusion.base import NoiseScheduler
from src.diffusion.beta_schedules import improved_beta_schedule
from src.model.unet import UNet
from src.utils.net import get_device, Device


def main():
    image_size = 28
    time_emb_dim = 112
    channels = 1
    num_diff_steps = 1000
    model_path = Path.cwd() / "models" / "uncond_unet_mnist.pth"

    dev = get_device(Device.GPU)
    unet = UNet(image_size, time_emb_dim, channels).to(dev)
    noise_scheduler = NoiseScheduler(improved_beta_schedule, num_diff_steps)
    unet.load_state_dict(th.load(model_path))

    samples = noise_scheduler.sample(unet, 100, dev, (1, 28, 28))[0]

    _, axes = plt.subplots(10, 10, figsize=(8, 8))
    for i in range(10):
        for j in range(10):
            axes[i, j].imshow(samples[i * 10 + j].squeeze(), cmap="gray")
            axes[i, j].axis("off")
    plt.show()


if __name__ == "__main__":
    main()
