"""Script for training a UNet-based unconditional diffusion model for MNIST"""


import sys


sys.path.append(".")
from pathlib import Path
import torch as th
import torch.nn.functional as F
import pytorch_lightning as pl
from src.data.cifar import get_cifar10_data_loaders
from src.diffusion.base import DiffusionSampler
from src.diffusion.beta_schedules import improved_beta_schedule
from src.model.imagenet import UNet, DiffusionModel, load_imagenet_diff_from_checkpoint
from src.utils.net import get_device, Device


def main():
    time_emb_dim = 112
    image_size = 32
    channels = 3
    num_diff_steps = 1000
    batch_size = 256
    dataloader_train, dataloader_val = get_cifar10_data_loaders(batch_size)

    model_path = Path.cwd() / "models" / "uncond_unet_cifar10.pt"
    if not model_path.parent.exists():
        print(f"Save dir. '{model_path.parent}' does not exist.")
        return

    dev = get_device(Device.GPU)
    if model_path.exists():
        print(f"Load existing model: {model_path.stem}")
        # unet = load_imagenet_diff_from_checkpoint(model_path, dev, image_size)
        raise NotImplementedError("Cifar10 chkpt not implemented yet.")
    else:
        unet = UNet(image_size, time_emb_dim, channels).to(dev)

    unet = UNet(image_size, time_emb_dim, channels).to(dev)
    unet.train()
    diff_sampler = DiffusionSampler(improved_beta_schedule, num_diff_steps)

    diffm = DiffusionModel(model=unet, loss_f=F.mse_loss, noise_scheduler=diff_sampler)

    diffm.to(dev)
    trainer = pl.Trainer(
        max_epochs=100,
        default_root_dir="logs/cifar10",
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        accelerator="gpu",
        devices=1,
    )

    trainer.fit(diffm, dataloader_train, dataloader_val)

    print("Saving model")
    th.save(unet.state_dict(), model_path)


if __name__ == "__main__":
    main()
