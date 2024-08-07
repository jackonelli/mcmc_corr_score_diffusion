"""Script for training a time-dependent classifier p(y | x_t, t).

For classifier-full guidance, we assume access to a likelihood function
p(y | x_t, t), which we estimate by a classifier which takes the diffusion step t as a parameter.
"""

import sys
import os

sys.path.append(".")
from argparse import ArgumentParser
from pathlib import Path

#
import torch as th
import pytorch_lightning as pl

#
from src.model.cifar.utils import select_classifier_t
from src.diffusion.base import DiffusionSampler
from src.model.trainers.classifier import (DiffusionClassifier, process_labelled_batch_cifar100,
                                           process_labelled_batch_cifar10)
from src.diffusion.beta_schedules import improved_beta_schedule, linear_beta_schedule, respaced_beta_schedule
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
        batch_fn = process_labelled_batch_cifar10
    elif args.dataset == "cifar100":
        dataloader_train, dataloader_val = get_cifar100_data_loaders(args.batch_size, data_root=args.dataset_path)
        num_classes, num_channels, img_size = CIFAR_100_NUM_CLASSES, CIFAR_NUM_CHANNELS, CIFAR_IMAGE_SIZE
        batch_fn = process_labelled_batch_cifar100
    else:
        raise ValueError('Invalid dataset')

    # Classifier
    class_t = select_classifier_t(arch=args.arch, dev=dev, num_channels=num_channels, img_size=img_size,
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
        batch_fn=batch_fn,
        batches_per_epoch=len(dataloader_train),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    diff_classifier.to(dev)

    ema = ''
    callbacks = []
    if args.ema:
        ema = '_ema'
        ema_callback = EMACallback(decay=0.9999)
        callbacks += [ema_callback]

    # lr_monitor = LearningRateMonitor(logging_interval="step")
    filename = (args.dataset + "_" + args.beta + "_" + args.arch + "_dropout" + str(int(args.dropout*100)) + ema +
                "weight_decay" + str(int(args.weight_decay*100)) + "_class_{epoch:02d}")
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


def parse_args():
    parser = ArgumentParser(prog="Train classification t model")
    parser.add_argument("--dataset", type=str, choices=["cifar100", "cifar10"], help="Dataset selection")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to dataset root")
    parser.add_argument("--max_epochs", type=int, default=-1, help="Max. number of epochs")
    parser.add_argument("--max_steps", type=int, default=int(3e5), help="Max. number of steps")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_diff_steps", type=int, default=1000, help="Number of time steps")
    parser.add_argument("--beta", type=str, choices=['lin', 'cos'], help='Beta schedule')
    parser.add_argument("--dropout", type=float, default=0., help="Dropout rate")
    parser.add_argument("--ema", action='store_true', help='If model is trained with EMA')
    parser.add_argument(
        "--arch", default="unet_ho_drop", type=str, choices=["unet", "resnet", "guided_diff", "unet_ho_drop"],
        help="Model architecture to use"
    )
    parser.add_argument(
        "--sim_batch", type=int, default=0, help="Simulation batch index, indexes parallell simulations."
    )
    parser.add_argument("--log_dir", default=None, help="Root directory for logging")
    parser.add_argument("--monitor", choices=['val_loss', 'train_loss'], default='val_loss',
                        help="Metric to monitor")
    parser.add_argument("--weight_decay", default=0, type=float, help="Weight decay")
    parser.add_argument("--learning_rate", default=2e-4, type=float, help="Learning rate")
    return parser.parse_args()


if __name__ == "__main__":
    main()
