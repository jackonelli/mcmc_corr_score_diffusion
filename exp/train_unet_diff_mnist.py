"""Script for training a UNet-based unconditional diffusion model for MNIST"""


from pathlib import Path
import torch as th
import torch.nn.functional as F
import pytorch_lightning as pl
from src.diffusion.base import DiffusionModel, NoiseScheduler
from src.diffusion.beta_schedules import improved_beta_schedule
from src.model.unet import UNet
from src.utils.net import get_device, Device
from src.data.mnist import get_mnist_data_loaders


def main():
    image_size = 28
    time_emb_dim = 112
    channels = 1
    num_diff_steps = 1000
    batch_size = 256

    dev = get_device(Device.GPU)
    unet = UNet(image_size, time_emb_dim, channels).to(dev)
    noise_scheduler = NoiseScheduler(improved_beta_schedule, num_diff_steps)

    diffm = DiffusionModel(model=unet, loss_f=F.mse_loss, noise_scheduler=noise_scheduler)

    diffm.to(dev)
    trainer = pl.Trainer(max_epochs=20, num_sanity_val_steps=0, accelerator="gpu", devices=1)

    dataloader_train, dataloader_val = get_mnist_data_loaders(batch_size)
    trainer.fit(diffm, dataloader_train, dataloader_val)

    print("Saving model")
    model_path = Path.cwd() / "models" / "uncond_unet_mnist.pth"
    th.save(unet.state_dict(), model_path)


if __name__ == "__main__":
    main()
