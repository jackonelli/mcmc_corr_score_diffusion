"""Classifier-full guidance sampling from multidimensional GMMs"""
import sys


sys.path.append(".")
from pathlib import Path
from argparse import ArgumentParser
import torch as th
from src.diffusion.base import DiffusionSampler
from src.diffusion.beta_schedules import (
    improved_beta_schedule,
)
from src.utils.net import get_device, Device
from src.model.comp_two_d.diffusion import ResnetDiffusionModel
from src.model.comp_two_d.classifier import load_classifier
from src.guidance.classifier_full import ClassifierFullGuidance
from src.guidance.base import GuidanceSampler, MCMCGuidanceSampler
from src.samplers.mcmc import (
    AnnealedHMCScoreSampler,
)
from src.utils.seeding import set_seed


def main():
    num_classes = 8
    guid_scale = 1.0
    args = parse_args()
    num_diff_steps = args.T
    set_seed(args.seed)

    # Setup and assign a directory where simulation results are saved.
    if args.low_rank_dim is None:
        sub_name = f"long_ep_full_rank_mean_sc_{args.mean_scale}"
    else:
        sub_name = f"long_ep_rank_{args.low_rank_dim}_{args.mean_scale}"

    sim_dir = Path.cwd() / f"results/multi_dim_gmm_T_{num_diff_steps}/{sub_name}"
    sim_dir.mkdir(exist_ok=True, parents=True)
    device = get_device(Device.GPU)

    models_dir = Path.cwd() / f"models/multi_dim_gmm_T_{num_diff_steps}/{sub_name}"

    betas = improved_beta_schedule(num_timesteps=num_diff_steps)
    time_steps = th.tensor([i for i in range(num_diff_steps)])
    diff_proc = DiffusionSampler(betas, time_steps, posterior_variance="beta")

    x_dim = args.dim
    diff_model = load_diff_model(models_dir / f"multi_dim_gmm_{x_dim}.pt", num_diff_steps, device, x_dim=x_dim)

    samples, _ = diff_proc.sample(diff_model, args.num_samples, device, th.Size((x_dim,)), verbose=True)
    samples = samples.detach().cpu()
    th.save(samples, sim_dir / f"samples_gmm_{x_dim}.th")


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
    parser.add_argument("--T", type=int, default=100, help="Number of diff. steps")
    parser.add_argument("--dim", type=int, default=10, help="x dim")
    parser.add_argument("--mcmc", type=str, default=None, help="MCMC method: {'la', 'hmc'}")
    parser.add_argument("--seed", type=int, default=None, help="Manual seed")
    parser.add_argument("--low_rank_dim", type=int, default=None, help="Low rank dim of GMMs")
    parser.add_argument("--mean_scale", type=float, default=1.0, help="Mean scale")
    parser.add_argument(
        "--sim_batch", type=int, default=0, help="Simulation batch index, indexes parallell simulations."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
