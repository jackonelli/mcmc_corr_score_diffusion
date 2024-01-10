"""Script for training a time-dependent classifier for simulated data"""


import sys


sys.path.append(".")
from argparse import ArgumentParser
from pathlib import Path
import json
import torch as th
import pytorch_lightning as pl
from src.data.comp_2d import GmmRadial
from src.data.utils import get_full_sample_data_loaders
from src.model.comp_two_d.classifier import Classifier
from src.diffusion.base import DiffusionSampler
from src.model.trainers.classifier import DiffusionClassifier
from src.diffusion.beta_schedules import improved_beta_schedule
from src.model.guided_diff.unet import initialise_diff_unet, load_pretrained_diff_unet
from src.utils.net import get_device, Device
from exp.utils import timestamp


def main():
    args = parse_args()
    # Diff params
    num_diff_steps = 100

    dataset = GmmRadial()
    dataloader_train, dataloader_val = get_full_sample_data_loaders(
        dataset=dataset, num_samples=100_000, batch_size=args.batch_size, num_val_samples=500
    )

    # Model params
    device = get_device(Device.GPU)
    diff_model = Classifier(x_dim=2, num_classes=dataset.num_comp, num_diff_steps=num_diff_steps)
    diff_model.to(device)
    diff_model.train()

    save_dir = _setup_results_dir(Path.cwd() / "models/train_comp_2d", args)

    betas = improved_beta_schedule(num_timesteps=num_diff_steps)
    time_steps = th.tensor([i for i in range(num_diff_steps)])
    diff_sampler = DiffusionSampler(betas, time_steps)

    diffm = DiffusionClassifier(model=diff_model, loss_f=th.nn.CrossEntropyLoss(), noise_scheduler=diff_sampler)
    # diffm.to(device)
    trainer = pl.Trainer(max_epochs=args.max_epochs, num_sanity_val_steps=1, accelerator="gpu", devices=1)

    trainer.fit(diffm, dataloader_train, dataloader_val)

    print("Saving model")
    th.save(diff_model.state_dict(), save_dir / f"class_t_gmm.pt")


def _setup_results_dir(res_dir: Path, args) -> Path:
    res_dir.mkdir(exist_ok=True)
    sim_dir = res_dir / f"class_t_gmm_{timestamp()}"
    sim_dir.mkdir(exist_ok=True)
    args_dict = vars(args)
    with open(sim_dir / "args.json", "w") as file:
        json.dump(args_dict, file, indent=2)

    return sim_dir


def parse_args():
    parser = ArgumentParser(prog="Sample from diffusion model")
    # parser.add_argument("--data", type=str, required=True, choices=["gmm", "bar"], help="Source dataset")
    parser.add_argument("--max_epochs", type=int, default=20, help="Max epochs")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size")
    parser.add_argument(
        "--sim_batch", type=int, default=0, help="Simulation batch index, indexes parallell simulations."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
