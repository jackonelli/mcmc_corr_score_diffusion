"""Script for training a UNet-based unconditional diffusion model for MNIST"""


from pathlib import Path
from argparse import ArgumentParser
import torch as th
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from src.diffusion.base import DiffusionSampler
from src.diffusion.trainer import DiffusionModel
from src.diffusion.beta_schedules import improved_beta_schedule
from src.model.unet import UNet, UNetEnergy
from src.utils.net import get_device, Device
from src.data.mnist import get_mnist_data_loaders


def main():
    args = parse_args()
    image_size = 28
    time_emb_dim = 112
    channels = 1
    num_diff_steps = 1000
    batch_size = 128
    model_type = ""
    if args.type == "energy":
        model_type = args.type
    model_path = Path.cwd() / "models" / f"{model_type}_{args.name}.pt"
    if not model_path.parent.exists():
        print(f"Save dir. '{model_path.parent}' does not exist.")
        return

    dev = get_device(Device.GPU)
    if args.type == "score":
        unet = UNet(image_size, time_emb_dim, channels).to(dev)
    else:
        unet = UNetEnergy(image_size, time_emb_dim, channels).to(dev)
    unet.train()
    betas = improved_beta_schedule(num_timesteps=num_diff_steps)
    time_steps = th.tensor([i for i in range(num_diff_steps)])
    diff_sampler = DiffusionSampler(betas, time_steps)

    diffm = DiffusionModel(model=unet, loss_f=F.mse_loss, noise_scheduler=diff_sampler)

    diffm.to(dev)

    checkpoint_callback = ModelCheckpoint(
        filename=args.name,
        save_last=True,
        every_n_epochs=1,
        save_top_k=1,
        monitor='val_loss'
    )

    if args.log_dir is None:
        root_dir = "logs/MNIST"
    else:
        root_dir = args.log_dir

    trainer = pl.Trainer(max_epochs=50,
                         num_sanity_val_steps=0,
                         default_root_dir=root_dir,
                         accelerator="gpu",
                         devices=1,
                         callbacks=[checkpoint_callback])

    dataloader_train, dataloader_val = get_mnist_data_loaders(batch_size, data_root=args.dataset_path)
    trainer.fit(diffm, dataloader_train, dataloader_val)

    if args.save:
        print(f"Saving model to '{model_path}'")
        th.save(unet.state_dict(), model_path)


def parse_args():
    parser = ArgumentParser(prog="Train unconditional diffusion model (score or energy)")
    parser.add_argument("--type", default="energy", type=str, choices=["score", "energy"])
    parser.add_argument("--name", default="uncond_unet_mnist", type=str)
    parser.add_argument("--log_dir", default=None, help="Root directory for logging")
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--dataset_path", default=None, help="Path to dataset root")
    return parser.parse_args()


if __name__ == "__main__":
    main()
