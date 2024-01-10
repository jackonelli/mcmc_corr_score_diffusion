"""Script for training a UNet-based unconditional diffusion model for MNIST"""


import sys


sys.path.append(".")
from argparse import ArgumentParser
from pathlib import Path
import json
import torch as th
import torch.nn.functional as F
import pytorch_lightning as pl
from src.data.comp_2d import Bar, GmmRadial
from src.data.utils import get_full_sample_data_loaders
from src.model.comp_two_d.diffusion import ResnetDiffusionModel, ResnetDiffusionModelEnergy
from src.diffusion.base import DiffusionSampler

# TODO: Move
from src.diffusion.trainer import DiffusionModel
from src.diffusion.beta_schedules import improved_beta_schedule
from src.utils.net import get_device, Device
from exp.utils import timestamp


def main():
    args = parse_args()
    # Diff params
    num_diff_steps = 100

    if args.data == "gmm":
        dataset = GmmRadial()
    elif args.data == "bar":
        dataset = Bar()
    dataloader_train, dataloader_val = get_full_sample_data_loaders(
        dataset=dataset, num_samples=100_000, batch_size=args.batch_size, num_val_samples=500
    )

    # Model params
    device = get_device(Device.GPU)
    # if args.load_weights:
    #     param_path = Path.cwd() / "models" / f"{args.load_weights}.pt"
    #     diff_model = load_pretrained_diff_unet(
    #         model_path=param_path, image_size=image_size, dev=device, class_cond=args.class_cond
    #     )
    #     diff_model.train()
    # else:
    #     diff_model = initialise_diff_unet(image_size=image_size, dev=device, class_cond=args.class_cond)
    #     diff_model.train()
    if args.energy:
        diff_model = ResnetDiffusionModelEnergy(num_diff_steps=num_diff_steps)
    else:
        diff_model = ResnetDiffusionModel(num_diff_steps=num_diff_steps)
    diff_model.to(device)
    diff_model.train()

    save_dir = _setup_results_dir(Path.cwd() / "models/train_comp_2d", args)

    betas = improved_beta_schedule(num_timesteps=num_diff_steps)
    time_steps = th.tensor([i for i in range(num_diff_steps)])
    diff_sampler = DiffusionSampler(betas, time_steps)

    diffm = DiffusionModel(model=diff_model, loss_f=F.mse_loss, noise_scheduler=diff_sampler)
    # diffm.to(device)
    trainer = pl.Trainer(max_epochs=args.max_epochs, num_sanity_val_steps=1, accelerator="gpu", devices=1)

    trainer.fit(diffm, dataloader_train, dataloader_val)

    model_name = 'score'
    if args.energy:
        model_name = 'energy'

    print("Saving model")
    th.save(diff_model.state_dict(), save_dir / f"2d_comp_{args.data}_{model_name}.pt")


def _setup_results_dir(res_dir: Path, args) -> Path:
    res_dir.mkdir(exist_ok=True)
    sim_dir = res_dir / f"{args.data}_{timestamp()}"
    sim_dir.mkdir(exist_ok=True)
    args_dict = vars(args)
    with open(sim_dir / "args.json", "w") as file:
        json.dump(args_dict, file, indent=2)

    return sim_dir


def parse_args():
    parser = ArgumentParser(prog="Sample from diffusion model")
    parser.add_argument("--data", type=str, required=True, choices=["gmm", "bar"], help="Source dataset")
    parser.add_argument("--max_epochs", type=int, default=200, help="Max epochs")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size")
    parser.add_argument("--energy", action="store_true", help='Use energy-parameterization instead of score')
    parser.add_argument(
        "--sim_batch", type=int, default=0, help="Simulation batch index, indexes parallell simulations."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
