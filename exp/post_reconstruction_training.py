"""Script for post-training a pre-trained ResNet-based classifier for MNIST

For reconstruction guidance, we assume access to a likelihood function
p(y | x_0), which we approximate with a ResNet-based classifier.
"""

from argparse import ArgumentParser
from pathlib import Path
import pytorch_lightning as pl
from src.model.resnet import load_classifier
from src.guidance.reconstruction import ReconstructionClassifier
from src.utils.net import get_device, Device
from src.data.mnist import get_mnist_data_loaders, get_noise_mnist_data_loader
import torch as th


def main():
    args = parse_args()
    model_path = Path.cwd() / "models" / "resnet_reconstruction_classifier_mnist.pt"
    device = get_device(Device.GPU)
    resnet = _load_class(model_path, device)
    resnet.train()

    diff_classifier = ReconstructionClassifier(model=resnet, loss_f=th.nn.CrossEntropyLoss(), lr=1e-10)
    diff_classifier.to(device)

    trainer = pl.Trainer(max_epochs=args.max_epochs, num_sanity_val_steps=0, accelerator="gpu", devices=1)

    _, dataloader_val = get_mnist_data_loaders(args.batch_size)
    dataloader_train = get_noise_mnist_data_loader(args.batch_size*20, args.batch_size)
    trainer.fit(diff_classifier, dataloader_train, dataloader_val)

    print("Saving model")
    model_path = Path.cwd() / "models" / "resnet_reconstruction_classifier_mnist_nt.pt"
    th.save(resnet.state_dict(), model_path)


def parse_args():
    parser = ArgumentParser(prog="Train reconstruction classifier")
    parser.add_argument("--max_epochs", default=1, type=int, help="Maximum number of epochs")
    parser.add_argument("--batch_size", default=500, type=int, help="Batch size")
    return parser.parse_args()


def _load_class(class_path: Path, device):
    classifier = load_classifier(class_path)
    classifier.to(device)
    classifier.eval()
    return classifier


if __name__ == "__main__":
    main()
