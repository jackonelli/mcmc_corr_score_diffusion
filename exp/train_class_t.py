"""Script for training a time-dependent classifier p(y | x_t, t).

For classifier-full guidance, we assume access to a likelihood function
p(y | x_t, t), which we estimate by a classifier which takes the diffusion step t as a parameter.
"""

import sys


sys.path.append(".")
from argparse import ArgumentParser
from pathlib import Path

#
import torch as th
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

#
from src.diffusion.base import DiffusionSampler
from src.model.trainers.classifier import DiffusionClassifier, process_labelled_batch_cifar100
from src.diffusion.beta_schedules import improved_beta_schedule, linear_beta_schedule, respaced_beta_schedule
from src.model.cifar.class_t import load_classifier_t as load_unet_classifier_t
from src.model.resnet import load_classifier_t as load_resnet_classifier_t
from src.model.cifar.unet_drop import load_classifier_t as load_unet_drop_classifier_t
from src.model.guided_diff.classifier import load_guided_classifier as load_guided_diff_classifier_t
from src.utils.net import get_device, Device
from src.data.cifar import (CIFAR_100_NUM_CLASSES, CIFAR_10_NUM_CLASSES, CIFAR_IMAGE_SIZE, CIFAR_NUM_CHANNELS,
                            get_cifar100_data_loaders, get_cifar10_data_loaders)
from src.utils.callbacks import EMACallback
from pytorch_lightning.callbacks import ModelCheckpoint


def main():
    args = parse_args()
    num_diff_steps = args.num_diff_steps
    model_path = Path.cwd() / "models" / f"{args.dataset}_{args.arch}_class_t.pt"
    if not model_path.parent.exists():
        print(f"Save dir. '{model_path.parent}' does not exist.")
        return
    dev = get_device(Device.GPU)

    # Data
    if args.dataset == "cifar10":
        dataloader_train, dataloader_val = get_cifar10_data_loaders(args.batch_size, data_root=args.dataset_path)
        num_classes, num_channels, img_size = CIFAR_10_NUM_CLASSES, CIFAR_NUM_CHANNELS, CIFAR_IMAGE_SIZE
    elif args.dataset == "cifar100":
        dataloader_train, dataloader_val = get_cifar100_data_loaders(args.batch_size, data_root=args.dataset_path)
        num_classes, num_channels, img_size = CIFAR_100_NUM_CLASSES, CIFAR_NUM_CHANNELS, CIFAR_IMAGE_SIZE
    else:
        raise ValueError('Invalid dataset')

    # Classifier
    class_t = select_classifier(args.arch, dev, num_channels=num_channels, img_size=img_size,
                                num_classes=num_classes, dropout=args.dropout, num_diff_steps=num_diff_steps)
    class_t.train()

    # Diffusion process
    if args.beta == 'lin':
        beta_schedule = linear_beta_schedule
    elif args.beta == 'cos':
        beta_schedule = improved_beta_schedule
    else:
        raise NotImplementedError("Not a valid beta schedule choice")
    betas, time_steps = respaced_beta_schedule(
        original_betas=beta_schedule(num_timesteps=num_diff_steps),
        T=num_diff_steps,
        respaced_T=num_diff_steps,
    )
    noise_scheduler = DiffusionSampler(betas, time_steps)

    diff_classifier = DiffusionClassifier(
        model=class_t,
        loss_f=th.nn.CrossEntropyLoss(),
        noise_scheduler=noise_scheduler,
        batch_fn=process_labelled_batch_cifar100,
        batches_per_epoch=len(dataloader_train),
    )
    diff_classifier.to(dev)

    ema = ''
    callbacks = []
    if args.ema:
        ema = '_ema'
        ema_callback = EMACallback(decay=0.9999)
        callbacks += [ema_callback]

    # lr_monitor = LearningRateMonitor(logging_interval="step")
    filename = args.dataset + "_" + args.beta + "_" + args.arch + "_" + str(int(args.dropout*100)) + ema + "_class_{epoch:02d}"
    checkpoint_callback = ModelCheckpoint(
        filename=filename,
        save_last=True,
        every_n_epochs=1,
        save_top_k=5,
        monitor=args.monitor
    )

    if args.log_dir is None:
        root_dir = "logs/" + args.dataset,
    else:
        root_dir = args.log_dir

    trainer = pl.Trainer(
        gradient_clip_val=1.,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        default_root_dir=root_dir,
        log_every_n_steps=100,
        num_sanity_val_steps=0,
        accelerator="gpu",
        devices=1,
        callbacks=[checkpoint_callback] + callbacks
    )

    trainer.fit(diff_classifier, dataloader_train, dataloader_val)

    print(f"Saving model to {model_path}")
    th.save(class_t.state_dict(), model_path)


def select_classifier(arch, dev, num_classes, num_channels, img_size, dropout=0., num_diff_steps=1000):
    if arch == "unet":
        class_t = load_unet_classifier_t(None, dev)
    elif arch == "resnet":
        class_t = load_resnet_classifier_t(
            model_path=None,
            dev=dev,
            emb_dim=256,
            num_classes=num_classes,
            num_channels=num_channels,
            dropout=dropout,
        ).to(dev)
    elif arch == "guided_diff":
        class_t = load_guided_diff_classifier_t(
            model_path=None, dev=dev, image_size=img_size, num_classes=num_classes,
            dropout=dropout,
        ).to(dev)
    elif arch == "unet_drop":
        x_size = (num_channels, img_size, img_size)
        class_t = load_unet_drop_classifier_t(model_path=None, dev=dev, dropout=dropout,
                                              num_diff_steps=num_diff_steps, num_classes=num_classes,
                                              x_size=x_size).to(dev)
    else:
        raise ValueError(f"Incorrect model arch: {arch}")
    return class_t


def parse_args():
    parser = ArgumentParser(prog="Train Cifar100 classification model")
    parser.add_argument("--dataset", type=str, choices=["cifar100", "cifar10"], help="Dataset selection")
    parser.add_argument("--dataset_path", type=Path, required=True, help="Path to dataset root")
    parser.add_argument("--max_epochs", type=int, default=-1, help="Max. number of epochs")
    parser.add_argument("--max_steps", type=int, default=int(8e5), help="Max. number of steps")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_diff_steps", type=int, default=1000, help="Number of time steps")
    parser.add_argument("--beta", type=str, choices=['lin', 'cos'], help='Beta schedule')
    parser.add_argument("--dropout", type=float, default=0., help="Dropout rate")
    parser.add_argument("--ema", action='store_true', help='If model is trained with EMA')
    parser.add_argument(
        "--arch", default="unet", type=str, choices=["unet", "resnet", "guided_diff", "unet_drop"],
        help="Model architecture to use"
    )
    parser.add_argument(
        "--sim_batch", type=int, default=0, help="Simulation batch index, indexes parallell simulations."
    )
    parser.add_argument("--log_dir", default=None, help="Root directory for logging")
    parser.add_argument("--monitor", choices=['val_loss', 'train_loss'], default='val_loss',
                        help="Metric to monitor")
    return parser.parse_args()


if __name__ == "__main__":
    main()
