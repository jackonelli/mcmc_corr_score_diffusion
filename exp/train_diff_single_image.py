"""Script for training a UNet-based unconditional diffusion model for MNIST"""


import sys


sys.path.append(".")
import os
from pathlib import Path
import torch as th
import torch.nn.functional as F
import pytorch_lightning as pl
from src.data.single_image import get_single_image_dataloader
from src.diffusion.base import DiffusionSampler
from src.model.imagenet import load_imagenet_diff
from src.diffusion.beta_schedules import improved_beta_schedule
from src.model.imagenet import UNet, DiffusionModel
from src.utils.net import get_device, Device


def main():
    image_size = 112  # 224
    time_emb_dim = 112
    channels = 3
    num_diff_steps = 1000
    dataloader_train, dataloader_val = get_single_image_dataloader(image_size)

    model_path = Path.cwd() / "models" / "single_image_diff.pt"
    if not model_path.parent.exists():
        print(f"Save dir. '{model_path.parent}' does not exist.")
        return

    dev = get_device(Device.GPU)
    if model_path.exists():
        print(f"Load existing model: {model_path.stem}")
        unet = load_imagenet_diff(model_path, dev, image_size)
    else:
        unet = UNet(image_size, time_emb_dim, channels).to(dev)
    unet.train()
    betas = improved_beta_schedule(num_timesteps=num_diff_steps)
    time_steps = th.tensor([i for i in range(num_diff_steps)])
    diff_sampler = DiffusionSampler(betas, time_steps)

    diffm = DiffusionModel(model=unet, loss_f=F.mse_loss, noise_scheduler=diff_sampler)

    diffm.to(dev)
    trainer = pl.Trainer(
        max_epochs=3,
        default_root_dir="logs/single_image",
        check_val_every_n_epoch=100,
        log_every_n_steps=100,
        num_sanity_val_steps=0,
        accelerator="gpu",
        devices=1,
    )

    trainer.fit(diffm, dataloader_train, dataloader_val)

    print("Saving model")
    th.save(unet.state_dict(), model_path)


if __name__ == "__main__":
    main()
