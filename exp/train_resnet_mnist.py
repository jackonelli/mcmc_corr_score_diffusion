"""Script for training a ResNet-based classifier for MNIST

For reconstruction guidance, we assume access to a likelihood function
p(y | x_0), which we approximate with a ResNet-based classifier.
"""

from argparse import ArgumentParser
from pathlib import Path
import torch as th
import pytorch_lightning as pl
from src.guidance.reconstruction import ReconstructionClassifier
from src.diffusion.beta_schedules import improved_beta_schedule
from src.model.resnet import ResNet, Bottleneck
from src.utils.net import get_device, Device
from src.data.mnist import get_mnist_data_loaders


def main():
    args = parse_args()
    model_path = Path.cwd() / "models" / "resnet_reconstruction_classifier_mnist.pt"
    if not model_path.parent.exists():
        print(f"Save dir. '{model_path.parent}' does not exist.")
        return

    dev = get_device(Device.GPU)

    resnet = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=10, num_channels=1).to(dev)
    resnet.train()

    diff_classifier = ReconstructionClassifier(model=resnet, loss_f=th.nn.CrossEntropyLoss())
    diff_classifier.to(dev)

    trainer = pl.Trainer(max_epochs=args.max_epochs, num_sanity_val_steps=0, accelerator="gpu", devices=1)

    dataloader_train, dataloader_val = get_mnist_data_loaders(args.batch_size)
    trainer.fit(diff_classifier, dataloader_train, dataloader_val)

    print("Saving model")
    th.save(resnet.state_dict(), model_path)


def parse_args():
    parser = ArgumentParser(prog="Train reconstruction classifier")
    parser.add_argument("--max_epochs", default=20, type=int, help="Maximum number of epochs")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size")
    return parser.parse_args()


if __name__ == "__main__":
    main()
