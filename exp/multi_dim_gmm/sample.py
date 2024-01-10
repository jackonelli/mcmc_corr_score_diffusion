"""Simulation script for cluster"""
import sys


sys.path.append(".")
from typing import Optional
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import torch as th
from src.diffusion.base import DiffusionSampler
from src.diffusion.beta_schedules import (
    improved_beta_schedule,
    respaced_beta_schedule,
)
from src.utils.net import get_device, Device
from src.model.comp_2d.diffusion import ResnetDiffusionModel
from src.model.comp_2d.classifier import load_classifier
from src.guidance.classifier_full import ClassifierFullGuidance
from src.guidance.base import GuidanceSampler, MCMCGuidanceSampler
from src.samplers.mcmc import (
    AnnealedHMCScoreSampler,
)
from src.utils.seeding import set_seed


def main():
    num_diff_steps = 100
    num_classes = 8
    guid_scale = 1.0
    args = parse_args()
    set_seed(args.seed)

    # Setup and assign a directory where simulation results are saved.
    sim_dir = Path.cwd() / "results/multi_dim_gmm"
    sim_dir.mkdir(exist_ok=True)
    device = get_device(Device.GPU)

    models_dir = Path.cwd() / "models/multi_dim_gmm"

    betas = improved_beta_schedule(num_timesteps=num_diff_steps)
    time_steps = th.tensor([i for i in range(num_diff_steps)])
    diff_proc = DiffusionSampler(betas, time_steps, posterior_variance="beta")

    start_dim, end_dim, num_steps = 10, 1000, 20
    x_dims = th.linspace(np.log(start_dim), np.log(end_dim), num_steps).exp().round().long()
    for i, x_dim in enumerate(x_dims, 1):
        print(f"x_dim: {x_dim}, {i} / {num_steps}")
        x_dim = int(x_dim.item())

        classes = th.randint(low=0, high=num_classes, size=(args.num_samples,)).long().to(device)
        diff_model = load_diff_model(models_dir / f"multi_dim_gmm_{x_dim}.pt", num_diff_steps, device, x_dim=x_dim)
        classifier = load_classifier(
            models_dir / f"class_t_gmm_{x_dim}.pt", num_classes, device, num_diff_steps=num_diff_steps, x_dim=x_dim
        )
        guidance = ClassifierFullGuidance(classifier, lambda_=guid_scale)
        guid_sampler = GuidanceSampler(diff_model, diff_proc, guidance, diff_cond=False)

        samples, _ = guid_sampler.sample(args.num_samples, classes, device, th.Size((x_dim,)), verbose=True)
        samples = samples.detach().cpu()
        th.save(samples, sim_dir / f"guid_samples_gmm_{x_dim}.th")

        if args.mcmc is not None:
            print("HMC guidance sampling")
            mcmc_steps = 10
            step_sizes = {int(t): 0.03 for t in range(0, num_diff_steps)}
            n_trapets = 5
            damping_coeff = 0.5
            leapfrog_steps = 3
            mcmc_sampler = AnnealedHMCScoreSampler(
                mcmc_steps, step_sizes, damping_coeff, th.ones_like(betas), leapfrog_steps, guidance.grad
            )
            guid_sampler = MCMCGuidanceSampler(
                diff_model=diff_model,
                diff_proc=diff_proc,
                guidance=guidance,
                mcmc_sampler=mcmc_sampler,
                reverse=True,
                diff_cond=False,
            )
            samples, _ = guid_sampler.sample(args.num_samples, classes, device, th.Size((x_dim,)), verbose=True)
            samples = samples.detach().cpu()
            th.save(samples, sim_dir / f"hmc_guid_samples_gmm_{x_dim}.th")
            print(f"Results written to '{sim_dir}'")
        classes = classes.detach().cpu()
        th.save(classes, sim_dir / f"guid_classes_gmm_{x_dim}.th")


def load_diff_model(diff_model_path, T, device, x_dim):
    assert diff_model_path.exists(), f"Model '{diff_model_path}' does not exist."
    diff_model = ResnetDiffusionModel(num_diff_steps=T, x_dim=x_dim)
    diff_model.load_state_dict(th.load(diff_model_path))
    diff_model.to(device)
    diff_model.eval()
    return diff_model


def parse_args():
    parser = ArgumentParser(prog="Sample from diffusion model")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of samples")
    parser.add_argument("--mcmc", type=str, default=None, help="MCMC method: {'la', 'hmc'}")
    parser.add_argument("--seed", type=Optional[int], default=None, help="Manual seed")
    parser.add_argument(
        "--sim_batch", type=int, default=0, help="Simulation batch index, indexes parallell simulations."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
