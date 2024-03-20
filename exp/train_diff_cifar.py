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
from src.model.cifar.unet_ho_drop import Unet_drop, UnetDropEnergy
from src.utils.net import get_device, Device
from pytorch_lightning.callbacks import ModelCheckpoint
from src.utils.callbacks import EMACallback
from src.model.cifar.utils import get_diff_model
from src.utils.file_mangement import find_num_trained_steps


def main():
    args = parse_args()
    if args.energy:
        param_model = "energy"
    else:
        param_model = "score"

    ema = ''
    callbacks = []
    if args.ema:
        ema = '_ema'
        path_model = None
        if args.path_checkpoint is not None:
            path_model = Path(args.path_checkpoint)
        ema_callback = EMACallback(decay=0.9999, path_ckpt=path_model)
        callbacks += [ema_callback]
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
                                          "_" + args.beta + "_" + str(int(args.dropout*100)) + ema + ".pt")
    if not model_path.parent.exists():
        print(f"Save dir. '{model_path.parent}' does not exist.")
        return

    dev = get_device(Device.GPU)
    if args.save and model_path.exists():
        raise NotImplementedError("This model already exists")
    else:
        if args.path_checkpoint is not None:
            path_model = Path(args.path_checkpoint)
            unet = get_diff_model(name=path_model.name,
                                  diff_model_path=path_model,
                                  device=dev,
                                  energy_param='energy' in path_model.name,
                                  image_size=image_size,
                                  num_steps=num_diff_steps,
                                  dropout=args.dropout,
                                  org_model=True)
        else:
            if args.model_size == 'small':
                if args.energy:
                    unet = UNetEnergy(image_size, time_emb_dim, channels, dropout=args.dropout).to(dev)
                else:
                    unet = UNet(image_size, time_emb_dim, channels, dropout=args.dropout).to(dev)
            elif args.model_size == 'large':
                if args.energy:
                    unet = UNetEnergy_Ho(dim=64, dim_mults=(1, 2, 4, 8), flash_attn=False, dropout=args.dropout)
                else:
                    unet = Unet_Ho(dim=64, dim_mults=(1, 2, 4, 8), flash_attn=False, dropout=args.dropout)
            elif args.model_size == 'large2':
                if args.energy:
                    unet = UnetDropEnergy(T=num_diff_steps, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
                                          num_res_blocks=2, dropout=args.dropout)
                else:
                    unet = Unet_drop(T=num_diff_steps, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
                                     num_res_blocks=2, dropout=args.dropout)
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

    diffm = DiffusionModel(model=unet, loss_f=F.mse_loss, noise_scheduler=diff_sampler, fixed=args.fixed_val,
                           path_load_state=args.path_checkpoint)
    filename = args.dataset + "_" + param_model + "_" + args.beta + "_" + args.model_size + "_" + str(int(args.dropout*100)) + ema + "_diff_{epoch:02d}"
    checkpoint_callback = ModelCheckpoint(
        filename=filename,
        save_last=True,
        every_n_epochs=1,
        save_top_k=2,
        monitor=args.monitor
    )

    diffm.to(dev)
    if args.log_dir is None:
        root_dir = "logs/" + args.dataset
    else:
        root_dir = args.log_dir

    max_steps = args.max_steps
    if args.path_checkpoint is not None:
        path_model = Path(args.path_checkpoint)
        max_steps = max_steps - find_num_trained_steps(path_model.name)

    trainer = pl.Trainer(
        gradient_clip_val=1.,
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        default_root_dir=root_dir,
        log_every_n_steps=100,
        num_sanity_val_steps=0,
        accelerator="gpu",
        devices=1,
        callbacks=[checkpoint_callback] + callbacks
    )

    trainer.fit(diffm, dataloader_train, dataloader_val)

    if args.save:
        print("Saving model")
        th.save(unet.state_dict(), model_path)

def parse_args():
    parser = ArgumentParser(prog="Train Cifar10 diffusion model")
    parser.add_argument("--energy", action='store_true', help="Use energy-parameterization")
    parser.add_argument("--model_size", choices=['small', 'large', 'large2'], help="Model size of Unet")
    parser.add_argument("--beta", choices=['lin', 'cos'], help="Type of beta schedule")
    parser.add_argument("--dataset", choices=['cifar10', 'cifar100'], help="Type of beta schedule")
    parser.add_argument("--dataset_path", default=None, help="Path to dataset root")
    parser.add_argument("--max_epochs", type=int, default=-1, help="Max. number of epochs")
    parser.add_argument("--max_steps", type=int, default=int(8e5), help="Max. number of steps")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--ema", action='store_true', help='If model is trained with EMA')
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout")
    parser.add_argument("--log_dir", default=None, help="Root directory for logging")
    parser.add_argument("--monitor", choices=['val_loss', 'train_loss'], default='val_loss',
                        help="Metric to monitor")
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--path_checkpoint", default=None, help="If path is provided then resume training "
                                                                "from checkpoint.")
    parser.add_argument("--fixed_val", action='store_true', help='If fixed val is True is that noise for validation always the same')
    return parser.parse_args()


if __name__ == "__main__":
    main()
