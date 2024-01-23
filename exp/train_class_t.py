"""Script for training a time-dependent classifier p(y | x_t, t).

For classifier-full guidance, we assume access to a likelihood function
p(y | x_t, t), which we estimate by a classifier which takes the diffusion step t as a parameter.
"""

import sys


sys.path.append(".")
from argparse import ArgumentParser
from pathlib import Path
import torch as th
import pytorch_lightning as pl
from src.diffusion.base import DiffusionSampler
from src.model.trainers.classifier import DiffusionClassifier, process_labelled_batch_cifar100
from src.diffusion.beta_schedules import improved_beta_schedule
from src.model.cifar.class_t import load_classifier_t as load_cifar_classifier_t
from src.model.resnet import load_classifier_t
from src.model.guided_diff.classifier import load_guided_classifier
from src.utils.net import get_device, Device
from src.data.cifar import CIFAR_100_NUM_CLASSES, CIFAR_IMAGE_SIZE, CIFAR_NUM_CHANNELS, get_cifar100_data_loaders


def main():
    args = parse_args()
    num_diff_steps = 1000
    batch_size = args.batch_size
    model_path = Path.cwd() / "models" / f"{args.dataset}_{args.arch}_class_t.pt"
    if not model_path.parent.exists():
        print(f"Save dir. '{model_path.parent}' does not exist.")
        return
    dev = get_device(Device.GPU)

    class_t = select_classifier(args.arch, dev)
    class_t.train()

    betas = improved_beta_schedule(num_timesteps=num_diff_steps)
    time_steps = th.tensor([i for i in range(num_diff_steps)])
    noise_scheduler = DiffusionSampler(betas, time_steps)

    diff_classifier = DiffusionClassifier(
        model=class_t,
        loss_f=th.nn.CrossEntropyLoss(),
        noise_scheduler=noise_scheduler,
        batch_fn=process_labelled_batch_cifar100,
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


def select_classifier(arch, dev):
    if arch == "unet":
        class_t = load_cifar_classifier_t(None, dev)
    elif arch == "resnet":
        class_t = load_classifier_t(
            model_path=None,
            dev=dev,
            emb_dim=112,
            num_classes=CIFAR_100_NUM_CLASSES,
            num_channels=CIFAR_NUM_CHANNELS,
        ).to(dev)
    elif arch == "guided_diff":
        class_t = load_guided_classifier(
            model_path=None, dev=dev, image_size=CIFAR_IMAGE_SIZE, num_classes=CIFAR_100_NUM_CLASSES
        ).to(dev)
    else:
        raise ValueError(f"Incorrect model arch: {arch}")
    return class_t


def parse_args():
    parser = ArgumentParser(prog="Train Cifar100 classification model")
    parser.add_argument("--dataset", type=str, choices=["cifar100", "mnist"], help="Dataset selection")
    parser.add_argument("--dataset_path", type=Path, required=True, help="Path to dataset root")
    parser.add_argument("--max_epochs", type=int, default=20, help="Max. number of epochs")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument(
        "--arch", default="unet", type=str, choices=["unet", "resnet", "guided_diff"], help="Model architecture to use"
    )
    parser.add_argument(
        "--sim_batch", type=int, default=0, help="Simulation batch index, indexes parallell simulations."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
