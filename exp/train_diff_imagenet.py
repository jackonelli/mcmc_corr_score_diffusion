"""Script for training a UNet-based unconditional diffusion model for MNIST"""


import sys

sys.path.append(".")
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple
import json
import torch as th
import torch.nn.functional as F
import pytorch_lightning as pl
from src.data.imagenet import get_imagenet_data_loaders
from src.diffusion.base import DiffusionSampler
from src.diffusion.trainer import LearnedVarDiffusion
from src.diffusion.beta_schedules import improved_beta_schedule
from src.model.guided_diff.unet import initialise_diff_unet, load_pretrained_diff_unet
from src.utils.net import get_device, Device
from exp.utils import timestamp


def main():
    args = parse_args()
    # Diff params
    num_diff_steps = 1000

    # Dataset params
    image_size = 128  # 224
    channels = 3
    batch_size = args.batch_size
    dataloader_train, dataloader_val = get_imagenet_data_loaders(
        Path.home() / "data/small-imagenet", image_size, batch_size
    )

    # Model params
    device = get_device(Device.GPU)
    if args.load_weights:
        param_path = Path.cwd() / "models" / f"{args.load_weights}.pt"
        diff_model = load_pretrained_diff_unet(
            model_path=param_path, image_size=image_size, dev=device, class_cond=args.class_cond
        )
        diff_model.train()
    else:
        diff_model = initialise_diff_unet(image_size=image_size, dev=device, class_cond=args.class_cond)
        diff_model.train()

    save_dir = _setup_results_dir(Path.cwd() / "models/train_imagenet", args)

    # if model_path.exists():
    #     print(f"Load existing model: {model_path.stem}")
    #     unet = load_imagenet_diff_from_checkpoint(model_path, device, image_size)
    # else:
    #     unet = UNet(image_size, time_emb_dim, channels).to(device)

    betas = improved_beta_schedule(num_timesteps=num_diff_steps)
    time_steps = th.tensor([i for i in range(num_diff_steps)])
    diff_sampler = DiffusionSampler(betas, time_steps)

    diffm = LearnedVarDiffusion(model=diff_model, loss_f=F.mse_loss, noise_scheduler=diff_sampler)

    # diffm.to(device)
    trainer = pl.Trainer(max_epochs=args.max_epochs, num_sanity_val_steps=1, accelerator="gpu", devices=1)

    trainer.fit(diffm, dataloader_train, dataloader_val)

    print("Saving model")
    th.save(diff_model.state_dict(), save_dir / "imagenet_diff_128.pt")


def _setup_results_dir(res_dir: Path, args) -> Path:
    res_dir.mkdir(exist_ok=True)
    model_type = "init" if not args.load_weights else "pretrained"
    sim_dir = res_dir / f"{model_type}_{timestamp()}"
    sim_dir.mkdir(exist_ok=True)
    args_dict = vars(args)
    with open(sim_dir / "args.json", "w") as file:
        json.dump(args_dict, file, indent=2)

    return sim_dir


def parse_args():
    parser = ArgumentParser(prog="Sample from diffusion model")
    parser.add_argument("--load_weights", type=str, help="Diffusion model file (without '.pt' extension)")
    parser.add_argument("--class_cond", action="store_true", help="Use class conditional diff. model")
    parser.add_argument("--max_epochs", type=int, default=2000, help="Max epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--sim_batch", type=int, default=0, help="Simulation batch index, indexes parallell simulations."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
