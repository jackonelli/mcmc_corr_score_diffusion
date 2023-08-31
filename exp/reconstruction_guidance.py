"""Script for sampling with reconstruction guidance"""


from pathlib import Path
from functools import partial
import torch as th
import torch.nn.functional as F
from src.model.resnet import load_classifier
from src.utils.net import Device, get_device
from src.diffusion.base import NoiseScheduler
from src.diffusion.beta_schedules import improved_beta_schedule
from src.model.unet import UNet
from src.data.mnist import mnist_transform


@th.no_grad()
def main():
    device = get_device(Device.GPU)
    models_dir = Path.cwd() / "models"
    uncond_diff = _load_diff(models_dir / "uncond_unet_mnist.pt", device)
    classifier = _load_class(models_dir / "resnet.pth.tar", device)


def likelihood(x_0, y, classifier):
    x_0 = mnist_transform(x_0)
    F.


def _load_diff(diff_path: Path, device):
    image_size = 28
    time_emb_dim = 112
    channels = 1
    unet = UNet(image_size, time_emb_dim, channels)
    unet.load_state_dict(th.load(diff_path))
    unet.to(device)
    unet.eval()
    return unet


def _load_class(class_path: Path, device):
    classifier = load_classifier(class_path)
    classifier.to(device)
    classifier.eval()
    return classifier


if __name__ == "__main__":
    main()
