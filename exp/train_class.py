"""Script for training a standard classifier p(y | x).

Used for debugging and also to train independent classifiers for metrics
"""

import sys


sys.path.append(".")
from argparse import ArgumentParser
from pathlib import Path
import torch as th
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from src.model.trainers.classifier import StandardClassifier, process_labelled_batch_cifar100
from src.model.cifar.class_t import load_classifier_t as load_cifar_classifier_t
from src.model.cifar.standard_class import StandardClassifier as SimpleCnn, load_standard_class
from src.model.resnet import load_classifier_t
from src.model.guided_diff.classifier import load_guided_classifier
from src.utils.net import get_device, Device
from src.data.cifar import CIFAR_100_NUM_CLASSES, CIFAR_IMAGE_SIZE, CIFAR_NUM_CHANNELS, get_cifar100_data_loaders


def main():
    args = parse_args()
    batch_size = args.batch_size
    model_path = Path.cwd() / "models" / f"{args.dataset}_{args.arch}_class.pt"
    if not model_path.parent.exists():
        print(f"Save dir. '{model_path.parent}' does not exist.")
        return
    dev = get_device(Device.GPU)

    # class_ = select_classifier(args.arch, dev)
    # class_ = SimpleCnn()
    class_ = load_standard_class(
        model_path=None,
        device=dev,
        num_channels=CIFAR_NUM_CHANNELS,
        num_classes=CIFAR_100_NUM_CLASSES,
    )
    class_.train()

    dataloader_train, dataloader_val = get_cifar100_data_loaders(batch_size, args.dataset_path)

    train_script = StandardClassifier(
        model=class_,
        loss_f=th.nn.CrossEntropyLoss(),
        batches_per_epoch=len(dataloader_train),
        batch_fn=process_labelled_batch_cifar100,
    )
    train_script.to(dev)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        num_sanity_val_steps=0,
        accelerator="gpu",
        gradient_clip_val=0.0,
        devices=1,
        default_root_dir="pl_logs/cifar100_class",
        callbacks=[lr_monitor],
    )

    trainer.fit(train_script, dataloader_train, dataloader_val)

    print(f"Saving model to {model_path}")
    th.save(class_.state_dict(), model_path)


def select_classifier(arch, dev):
    if arch == "unet":
        class_t = load_cifar_classifier_t(None, dev, time_emb_dim=1)
    elif arch == "resnet":
        class_t = load_classifier_t(
            model_path=None,
            dev=dev,
            emb_dim=1,
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
        "--arch",
        default="unet",
        type=str,
        choices=["unet", "resnet", "guided_diff", "simple"],
        help="Model architecture to use",
    )
    parser.add_argument(
        "--sim_batch", type=int, default=0, help="Simulation batch index, indexes parallell simulations."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
