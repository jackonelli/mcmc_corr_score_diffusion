"""Script for training a UNet-based unconditional diffusion model for MNIST"""


import sys


sys.path.append(".")
from pathlib import Path
from argparse import ArgumentParser
import torch as th
import torch.nn.functional as F
import pytorch_lightning as pl
from src.data.cifar import get_cifar10_data_loaders, get_cifar100_data_loaders
from src.diffusion.base import DiffusionSampler
from src.diffusion.beta_schedules import improved_beta_schedule, linear_beta_schedule, respaced_beta_schedule
from src.model.cifar.unet import UNet, UNetEnergy
from src.model.trainers.diffusion import DiffusionModel
from src.model.cifar.unet_ho import Unet_Ho, UNetEnergy_Ho
from src.utils.net import get_device, Device
from pytorch_lightning.callbacks import ModelCheckpoint
from src.utils.callbacks import EMACallback


def main():
    args = parse_args()
    if args.energy:
        param_model = "energy"
    else:
        param_model = "score"
    time_emb_dim = 112
    image_size = 32
    channels = 3
    num_diff_steps = 1000
    if args.dataset == "cifar10":
        dataloader_train, dataloader_val = get_cifar10_data_loaders(args.batch_size, data_root=args.dataset_path)
    elif args.dataset == "cifar100":
        dataloader_train, dataloader_val = get_cifar100_data_loaders(args.batch_size, data_root=args.dataset_path)
    else:
        raise ValueError('Invalid dataset')

    model_path = Path.cwd() / "models" / (param_model + "_uncond_unet_" + args.dataset + "_" + args.model_size +
                                          "_" + args.beta + ".pt")
    if not model_path.parent.exists():
        print(f"Save dir. '{model_path.parent}' does not exist.")
        return

    dev = get_device(Device.GPU)
    if model_path.exists():
        raise NotImplementedError("This model already exists")
    else:
        if args.model_size == 'small':
            if args.energy:
                unet = UNetEnergy(image_size, time_emb_dim, channels).to(dev)
            else:
                unet = UNet(image_size, time_emb_dim, channels).to(dev)
        elif args.model_size == 'large':
            if args.energy:
                unet = UNetEnergy_Ho(dim=64, dim_mults=(1, 2, 4, 8), flash_attn=False)
            else:
                unet = Unet_Ho(dim=64, dim_mults=(1, 2, 4, 8), flash_attn=False)
        else:
            raise NotImplementedError("Not a valid model size.")

    if args.beta == 'lin':
        beta_schedule, post_var = linear_beta_schedule, "beta"
    elif args.beta == 'cos':
        beta_schedule, post_var = improved_beta_schedule, "beta"
    else:
        raise NotImplementedError("Not a valid beta schedule choice")

    betas, time_steps = respaced_beta_schedule(
        original_betas=beta_schedule(num_timesteps=num_diff_steps),
        T=num_diff_steps,
        respaced_T=num_diff_steps,
    )

    unet.train()
    diff_sampler = DiffusionSampler(betas, time_steps)

    diffm = DiffusionModel(model=unet, loss_f=F.mse_loss, noise_scheduler=diff_sampler)
    filename = args.dataset + "_" + param_model + "_" + args.beta + "_" + args.model_size + "_diff_{epoch:02d}"
    checkpoint_callback = ModelCheckpoint(
        filename=filename,
        every_n_epochs=1,
        save_top_k=5,
        monitor='val_loss'
    )
    ema_callback = EMACallback(decay=0.9999)

    diffm.to(dev)
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        default_root_dir="logs/" + args.dataset,
        log_every_n_steps=100,
        num_sanity_val_steps=0,
        accelerator="gpu",
        devices=1,
        callbacks=[checkpoint_callback, ema_callback]
    )

    trainer.fit(diffm, dataloader_train, dataloader_val)

    print("Saving model")
    th.save(unet.state_dict(), model_path)

def parse_args():
    parser = ArgumentParser(prog="Train Cifar10 diffusion model")
    parser.add_argument("--energy", action='store_true', help="Use energy-parameterization")
    parser.add_argument("--model_size", choices=['small', 'large'], help="Model size of Unet")
    parser.add_argument("--beta", choices=['lin', 'cos'], help="Type of beta schedule")
    parser.add_argument("--dataset", choices=['cifar10', 'cifar100'], help="Type of beta schedule")
    parser.add_argument("--dataset_path", type=Path, required=True, help="Path to dataset root")
    parser.add_argument("--max_epochs", type=int, default=-1, help="Max. number of epochs")
    parser.add_argument("--max_steps", type=int, default=int(8e5), help="Max. number of steps")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    return parser.parse_args()


if __name__ == "__main__":
    main()
