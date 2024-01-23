"""Script for training a ResNet-based classifier for x_t.

For classifier-full guidance, we assume access to a likelihood function
p(y | x_t, t), which we estimate by a ResNet-based classifier which takes the diffusion step t as a parameter.
"""

import sys


sys.path.append(".")
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import torch as th
import pytorch_lightning as pl
from src.diffusion.base import DiffusionSampler
from src.model.trainers.classifier import DiffusionClassifier, _process_labelled_batch_cifar100
from src.diffusion.beta_schedules import improved_beta_schedule
from src.model.resnet import load_classifier_t
from src.utils.net import get_device, Device
from src.data.cifar import CIFAR_NUM_CHANNELS, get_cifar100_data_loaders, CIFAR_IMAGE_SIZE, CIFAR_100_NUM_CLASSES


def main():
    args = parse_args()
    n_classes = CIFAR_100_NUM_CLASSES
    time_emb_dim = 112
    num_diff_steps = 1000
    batch_size = args.batch_size
    model_path = Path.cwd() / "models" / "{args.dataset}.pt"
    if not model_path.parent.exists():
        print(f"Save dir. '{model_path.parent}' does not exist.")
        return

    dev = get_device(Device.GPU)

    class_t = load_classifier_t(
        model_path=None,
        dev=dev,
        emb_dim=112,
        num_classes=CIFAR_100_NUM_CLASSES,
        num_channels=CIFAR_NUM_CHANNELS,
    ).to(dev)
    class_t.train()

    betas = improved_beta_schedule(num_timesteps=num_diff_steps)
    time_steps = th.tensor([i for i in range(num_diff_steps)])
    noise_scheduler = DiffusionSampler(betas, time_steps)

    diff_classifier = DiffusionClassifier(
        model=class_t,
        loss_f=th.nn.CrossEntropyLoss(),
        noise_scheduler=noise_scheduler,
        batch_fn=_process_labelled_batch_cifar100,
    )
    diff_classifier.to(dev)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        num_sanity_val_steps=0,
        accelerator="gpu",
        devices=1,
        default_root_dir="pl_logs/cifar100_class_t",
    )

    dataloader_train, dataloader_val = get_cifar100_data_loaders(batch_size, args.dataset_path)
    trainer.fit(diff_classifier, dataloader_train, dataloader_val)

    print(f"Saving model to {model_path}")
    th.save(class_t.state_dict(), model_path)


@dataclass()
class ClassifierHyperParams:
    pass


def parse_args():
    parser = ArgumentParser(prog="Train Cifar100 diffusion model")
    parser.add_argument("--dataset", type=str, choices=["cifar100", "mnist"], help="Dataset selection")
    parser.add_argument("--dataset_path", type=Path, required=True, help="Path to dataset root")
    parser.add_argument("--max_epochs", type=int, default=20, help="Max. number of epochs")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("--type", default="score", type=str, choices=["score", "energy"])
    parser.add_argument(
        "--sim_batch", type=int, default=0, help="Simulation batch index, indexes parallell simulations."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
