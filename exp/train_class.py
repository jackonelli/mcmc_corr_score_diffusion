"""Script for training a standard classifier p(y | x).

Used for debugging and also to train independent classifiers for metrics
"""

import sys


sys.path.append(".")
from argparse import ArgumentParser
from pathlib import Path
import torch as th
import os
import pytorch_lightning as pl
from src.model.trainers.classifier import StandardClassifier
from src.model.cifar.utils import select_classifier
from src.utils.callbacks import EMACallback
from src.utils.net import get_device, Device
from src.data.cifar import (CIFAR_100_NUM_CLASSES, CIFAR_10_NUM_CLASSES, CIFAR_IMAGE_SIZE, CIFAR_NUM_CHANNELS,
                            get_cifar100_data_loaders, get_cifar10_data_loaders)
from src.model.trainers.classifier import (process_labelled_batch_cifar100,
                                           process_labelled_batch_cifar10)
from pytorch_lightning.callbacks import ModelCheckpoint


def main():
    args = parse_args()
    model_path = Path.cwd() / "models" / f"{args.dataset}_{args.arch}_class.pt"
    if args.save and not model_path.parent.exists():
        print(f"Save dir. '{model_path.parent}' does not exist.")
        return
    dev = get_device(Device.GPU)

    # Data
    if args.dataset == "cifar10":
        dataloader_train, dataloader_val = get_cifar10_data_loaders(args.batch_size, data_root=args.dataset_path)
        num_classes, num_channels, img_size = CIFAR_10_NUM_CLASSES, CIFAR_NUM_CHANNELS, CIFAR_IMAGE_SIZE
        batch_fn = process_labelled_batch_cifar10
    elif args.dataset == "cifar100":
        dataloader_train, dataloader_val = get_cifar100_data_loaders(args.batch_size, data_root=args.dataset_path)
        num_classes, num_channels, img_size = CIFAR_100_NUM_CLASSES, CIFAR_NUM_CHANNELS, CIFAR_IMAGE_SIZE
        batch_fn = process_labelled_batch_cifar100
    else:
        raise ValueError('Invalid dataset')

    classifier = select_classifier(args.arch, dev, num_classes, num_channels, args.dataset)
    classifier.train()

    train_script = StandardClassifier(
        model=classifier,
        loss_f=th.nn.CrossEntropyLoss(),
        batches_per_epoch=len(dataloader_train),
        batch_fn=batch_fn,
    )
    train_script.to(dev)

    ema = ''
    callbacks = []
    if args.ema:
        ema = '_ema'
        ema_callback = EMACallback(decay=0.9999)
        callbacks += [ema_callback]

    filename = (args.dataset + "_" + args.arch + ema + "_class_{epoch:02d}")
    checkpoint_callback = ModelCheckpoint(
        filename=filename,
        save_last=True,
        every_n_epochs=1,
        save_top_k=5,
        monitor=args.monitor
    )

    if args.log_dir is None:
        root_dir = os.path.join('logs', args.dataset)
    else:
        root_dir = args.log_dir

    trainer = pl.Trainer(
        gradient_clip_val=1.,
        max_steps=args.max_steps,
        default_root_dir=root_dir,
        log_every_n_steps=100,
        num_sanity_val_steps=0,
        accelerator="gpu",
        devices=1,
        callbacks=[checkpoint_callback] + callbacks
    )

    trainer.fit(train_script, dataloader_train, dataloader_val)

    if args.save:
        print("Saving model")
        th.save(classifier.state_dict(), model_path)


def parse_args():
    parser = ArgumentParser(prog="Train classification model")
    parser.add_argument("--dataset", type=str, choices=["cifar100", "cifar10"], help="Dataset selection")
    parser.add_argument("--dataset_path", default=None, help="Path to dataset root")
    parser.add_argument("--max_steps", type=int, default=8e5, help="Max. number of steps")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--ema", action='store_true', help='If model is trained with EMA')
    parser.add_argument(
        "--arch",
        default="vgg13_bn",
        type=str,
        choices=["simple_resnet", "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"],
        help="Model architecture to use",
    )
    parser.add_argument(
        "--sim_batch", type=int, default=0, help="Simulation batch index, indexes parallell simulations."
    )
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--monitor", choices=['val_loss', 'train_loss'], default='val_loss',
                     help="Metric to monitor")
    parser.add_argument("--log_dir", default=None, help="Root directory for logging")
    return parser.parse_args()


if __name__ == "__main__":
    main()