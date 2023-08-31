"""Script for sampling with reconstruction guidance"""


from pathlib import Path
import torch as th
from src.model.resnet import load_classifier
from src.data.mnist import get_mnist_data_loaders
from src.utils.net import Device, get_device


@th.no_grad()
def main():
    device = get_device(Device.GPU)
    models_dir = Path.cwd() / "models"
    classifier = load_classifier(models_dir / "models/resnet.pth.tar")
    classifier.to(device)
    classifier.eval()


if __name__ == "__main__":
    main()
