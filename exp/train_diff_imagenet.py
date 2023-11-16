"""Script for training a UNet-based unconditional diffusion model for MNIST"""


import sys
sys.path.append(".")
from pathlib import Path
import torch as th
import torch.nn.functional as F
import pytorch_lightning as pl
from src.data.imagenet import get_imagenet_data_loaders
from src.diffusion.base import DiffusionSampler
from src.diffusion.beta_schedules import improved_beta_schedule
from src.model.imagenet import UNet, DiffusionModel
from src.utils.net import get_device, Device


def main():
    image_size = 112 #224
    time_emb_dim = 112
    channels = 3
    num_diff_steps = 1000
    batch_size = 64
    dataloader_train, dataloader_val = get_imagenet_data_loaders(Path("/data/small-imagenet"), image_size, batch_size)

    model_path = Path.cwd() / "models" / "uncond_unet_imagenet.pt"
    if not model_path.parent.exists():
        print(f"Save dir. '{model_path.parent}' does not exist.")
        return

    dev = get_device(Device.GPU)
    unet = UNet(image_size, time_emb_dim, channels).to(dev)
    unet.train()
    diff_sampler = DiffusionSampler(improved_beta_schedule, num_diff_steps)

    diffm = DiffusionModel(model=unet, loss_f=F.mse_loss, noise_scheduler=diff_sampler)

    diffm.to(dev)
    trainer = pl.Trainer(max_epochs=20, num_sanity_val_steps=0, accelerator="gpu", devices=1)

    trainer.fit(diffm, dataloader_train, dataloader_val)

    print("Saving model")
    model_path = Path.cwd() / "models" / "imagenet_diff.pt"
    th.save(unet.state_dict(), model_path)


if __name__ == "__main__":
    main()
