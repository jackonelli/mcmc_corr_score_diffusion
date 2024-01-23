"""Script for training a UNet-based unconditional diffusion model for CIFAR-100"""


import sys


sys.path.append(".")
from argparse import ArgumentParser
from pathlib import Path
import torch as th
import torch.nn.functional as F
import pytorch_lightning as pl
from src.data.cifar import get_cifar100_data_loaders
from src.diffusion.base import DiffusionSampler
from src.diffusion.beta_schedules import improved_beta_schedule, respaced_beta_schedule
from src.model.cifar.unet import UNet, UNetEnergy
from src.model.trainers.diffusion import DiffusionModel
from src.utils.net import get_device, Device


def main():
    time_emb_dim = 112
    image_size = 32
    channels = 3
    args = parse_args()
    batch_size = args.batch_size
    dataloader_train, dataloader_val = get_cifar100_data_loaders(batch_size, data_root=args.dataset_path)

    dev = get_device(Device.GPU)
    if args.type == "score":
        unet = UNet(image_size, time_emb_dim, channels).to(dev)
    else:
        unet = UNetEnergy(image_size, time_emb_dim, channels).to(dev)

    model_path = Path.cwd() / "models" / f"{args.type}_uncond_unet_cifar100.pt"
    if not model_path.parent.exists():
        print(f"Save dir. '{model_path.parent}' does not exist.")
        return

    unet.train()
    num_diff_steps = 1000
    post_var = "beta"
    betas, time_steps = respaced_beta_schedule(
        original_betas=improved_beta_schedule(num_timesteps=num_diff_steps),
        T=num_diff_steps,
        respaced_T=num_diff_steps,
    )
    diff_sampler = DiffusionSampler(betas, time_steps, posterior_variance=post_var)
    diffm = DiffusionModel(model=unet, loss_f=F.mse_loss, noise_scheduler=diff_sampler)

    diffm.to(dev)
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        default_root_dir="pl_logs/cifar100_diff",
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        accelerator="gpu",
        devices=1,
    )

    trainer.fit(diffm, dataloader_train, dataloader_val)

    print("Saving model")
    th.save(unet.state_dict(), model_path)


def parse_args():
    parser = ArgumentParser(prog="Train Cifar100 diffusion model")
    parser.add_argument("--dataset_path", type=Path, required=True, help="Path to dataset root")
    parser.add_argument("--max_epochs", type=int, default=int(1e5), help="Max. number of epochs")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
    parser.add_argument("--type", default="score", type=str, choices=["score", "energy"])
    parser.add_argument(
        "--sim_batch", type=int, default=0, help="Simulation batch index, indexes parallell simulations."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
