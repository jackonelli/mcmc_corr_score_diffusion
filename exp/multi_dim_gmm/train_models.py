"""Script for training diffusion and classifier models for varying input dimension"""


import sys


sys.path.append(".")
from argparse import ArgumentParser
from pathlib import Path
import json
import numpy as np
import torch as th
import torch.nn.functional as F
import pytorch_lightning as pl
from src.data.multi_dim_gmm import Gmm, generate_means, threshold_covs
from src.data.utils import get_full_sample_data_loaders
from src.model.comp_2d.diffusion import ResnetDiffusionModel
from src.diffusion.base import DiffusionSampler

# TODO: Move
from src.diffusion.trainer import DiffusionModel
from src.model.comp_2d.classifier import Classifier
from src.model.trainers.classifier import DiffusionClassifier
from src.diffusion.beta_schedules import improved_beta_schedule
from src.utils.net import get_device, Device


def main():
    args = parse_args()
    device = get_device(Device.GPU)
    save_dir = _setup_results_dir(Path.cwd() / "models/multi_dim_gmm", args)
    # Diff params
    num_diff_steps = 100
    betas = improved_beta_schedule(num_timesteps=num_diff_steps)
    time_steps = th.tensor([i for i in range(num_diff_steps)])
    diff_sampler = DiffusionSampler(betas, time_steps)

    num_comp, low_rank_dim = 8, 10

    start_dim, end_dim, num_steps = 10, 1000, 20
    # Minimum distance between means is sqrt(2)
    std = float(np.sqrt(2)) / 4
    x_dims = th.linspace(np.log(start_dim), np.log(end_dim), num_steps).exp().round().long()
    for i, x_dim in enumerate(x_dims, 1):
        x_dim = int(x_dim.item())
        print(f"x_dim: {x_dim}, {i} / {num_steps}")
        means = generate_means(x_dim, num_comp)
        covs = [threshold_covs(x_dim, low_rank_dim, std**2) for _ in range(num_comp)]
        dataset = Gmm(means, covs)
        dataloader_train, dataloader_val = get_full_sample_data_loaders(
            dataset=dataset, num_samples=100_000, batch_size=args.batch_size, num_val_samples=500
        )

        # Diff model
        diff_model = ResnetDiffusionModel(num_diff_steps=num_diff_steps, x_dim=x_dim)
        diff_model.to(device)
        diff_model.train()
        diffm = DiffusionModel(model=diff_model, loss_f=F.mse_loss, noise_scheduler=diff_sampler)
        trainer = pl.Trainer(max_epochs=args.max_epochs, num_sanity_val_steps=1, accelerator="gpu", devices=1)
        trainer.fit(diffm, dataloader_train, dataloader_val)
        th.save(diff_model.state_dict(), save_dir / f"multi_dim_gmm_{x_dim}.pt")

        # Classifier model
        classifier = Classifier(x_dim=x_dim, num_classes=num_comp, num_diff_steps=num_diff_steps)
        classifier.to(device)
        classifier.train()
        classifier_trainer = DiffusionClassifier(
            model=classifier, loss_f=th.nn.CrossEntropyLoss(), noise_scheduler=diff_sampler
        )
        trainer = pl.Trainer(max_epochs=args.max_epochs, num_sanity_val_steps=1, accelerator="gpu", devices=1)

        trainer.fit(classifier_trainer, dataloader_train, dataloader_val)
        th.save(classifier.state_dict(), save_dir / f"class_t_gmm_{x_dim}.pt")


def _setup_results_dir(sim_dir: Path, args) -> Path:
    sim_dir.mkdir(exist_ok=True)
    args_dict = vars(args)
    with open(sim_dir / "args.json", "w") as file:
        json.dump(args_dict, file, indent=2)

    return sim_dir


def parse_args():
    parser = ArgumentParser(prog="Sample from diffusion model")
    parser.add_argument("--max_epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size")
    parser.add_argument(
        "--sim_batch", type=int, default=0, help="Simulation batch index, indexes parallell simulations."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
