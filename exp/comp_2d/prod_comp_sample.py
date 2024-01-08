"""Simulation script for cluster"""
import sys


sys.path.append(".")
from typing import Optional
from pathlib import Path
from argparse import ArgumentParser
import torch as th
from src.diffusion.base import DiffusionSampler
from src.diffusion.beta_schedules import (
    improved_beta_schedule,
    respaced_beta_schedule,
)
from src.utils.net import get_device, Device
from src.model.comp_2d.diffusion import ResnetDiffusionModel
from src.samplers.mcmc import (
    AnnealedHMCScoreSampler,
    AnnealedLAScoreSampler,
    AnnealedUHMCScoreSampler,
    AnnealedULAScoreSampler,
)
from src.comp.base import ProductCompSampler
from exp.utils import SimulationConfig, setup_results_dir, get_step_size
from src.utils.seeding import set_seed


def main():
    T = 100
    args = parse_args()
    set_seed(args.seed)

    # Setup and assign a directory where simulation results are saved.
    sim_dir = Path.cwd() / "results/comp_2d"
    sim_dir.mkdir(exist_ok=True)
    device = get_device(Device.GPU)

    models_dir = Path.cwd() / "models/comp_2d"
    diff_model_gmm = load_diff_model(models_dir / "gmm.pt", T, device)
    diff_model_bar = load_diff_model(models_dir / "bar.pt", T, device)

    betas, time_steps = respaced_beta_schedule(
        original_betas=improved_beta_schedule(num_timesteps=T),
        T=T,
        respaced_T=T,
    )
    diff_proc = DiffusionSampler(betas, time_steps, posterior_variance="beta")
    prod_sampler = ProductCompSampler(diff_model_gmm, diff_model_bar, diff_proc, use_reverse_step=args.use_rev)

    if args.sample_separate:
        print("Sampling from GMM model")
        gmm_samples, _ = diff_proc.sample(diff_model_gmm, args.num_samples, device, th.Size((2,)), verbose=True)
        gmm_samples = gmm_samples.detach().cpu()
        th.save(gmm_samples, sim_dir / f"gmm_samples.th")
        print("Sampling from bar model")
        bar_samples, _ = diff_proc.sample(diff_model_bar, args.num_samples, device, th.Size((2,)), verbose=True)
        bar_samples = bar_samples.detach().cpu()
        th.save(bar_samples, sim_dir / f"bar_samples.th")
    if args.mcmc is None:
        print("Sampling from product model with reverse process")
        prod_samples, _ = prod_sampler.sample(args.num_samples, device, th.Size((2,)), verbose=True)
        prod_samples = prod_samples.detach().cpu()
        th.save(prod_samples, sim_dir / f"prod_samples_reverse.th")
    elif args.mcmc == "hmc":
        print("Sampling from product model")
        mcmc_steps = 10
        step_sizes = {int(t): 0.03 for t in range(0, T)}
        n_trapets = 5
        damping_coeff = 0.5
        leapfrog_steps = 3
        mcmc_sampler = AnnealedHMCScoreSampler(
            mcmc_steps, step_sizes, damping_coeff, th.ones_like(betas), leapfrog_steps, prod_sampler.grad
        )
        prod_samples, _ = prod_sampler.mcmc_sample(args.num_samples, mcmc_sampler, device, th.Size((2,)), verbose=True)
        prod_samples = prod_samples.detach().cpu()
        th.save(
            prod_samples, sim_dir / f"prod_samples_{args.mcmc}_{'with' if prod_sampler.reverse else 'without'}_rev.th"
        )
    else:
        pass
    print(f"Results written to '{sim_dir}'")


def load_diff_model(diff_model_path, T, device):
    assert diff_model_path.exists(), f"Model '{diff_model_path}' does not exist."
    diff_model = ResnetDiffusionModel(num_diff_steps=T)
    diff_model.load_state_dict(th.load(diff_model_path))
    diff_model.to(device)
    diff_model.eval()
    return diff_model


def parse_args():
    parser = ArgumentParser(prog="Sample from diffusion model")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of samples")
    parser.add_argument("--mcmc", type=str, default=None, help="MCMC method: {'la', 'hmc'}")
    parser.add_argument("--use_rev", action="store_true", help="Use reverse step")
    parser.add_argument("--sample_separate", action="store_true", help="Use reverse step")
    parser.add_argument("--seed", type=Optional[int], default=None, help="Manual seed")
    parser.add_argument(
        "--sim_batch", type=int, default=0, help="Simulation batch index, indexes parallell simulations."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
