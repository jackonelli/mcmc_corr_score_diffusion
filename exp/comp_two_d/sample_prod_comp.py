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
from src.model.comp_two_d.diffusion import ResnetDiffusionModel
from src.samplers.mcmc import (
    AnnealedHMCEnergySampler,
    AnnealedHMCScoreSampler,
    AnnealedLAEnergySampler,
    AnnealedLAScoreSampler,
    AnnealedUHMCEnergySampler,
    AnnealedUHMCScoreSampler,
    AnnealedULAEnergySampler,
    AnnealedULAScoreSampler,
    MCMCMHCorrSampler,
)
from src.comp.base import ProductCompSampler
from exp.comp_two_d.utils import SimulationConfig, setup_results_dir
from src.utils.seeding import set_seed


def main():
    args = parse_args()
    config = SimulationConfig.from_json(args.config)
    T = config.num_diff_steps
    set_seed(config.seed)

    # Setup and assign a directory where simulation results are saved.
    sim_dir = setup_results_dir(config)
    sim_dir.mkdir(exist_ok=True)
    device = get_device(Device.GPU)

    models_dir = Path.cwd() / "models/comp_two_d"
    diff_model_gmm = load_diff_model(models_dir / config.diff_model_1, T, device)
    diff_model_bar = load_diff_model(models_dir / config.diff_model_2, T, device)

    betas, time_steps = respaced_beta_schedule(
        original_betas=improved_beta_schedule(num_timesteps=T),
        T=T,
        respaced_T=T,
    )
    diff_proc = DiffusionSampler(betas, time_steps, posterior_variance="beta")

    prod_sampler, mcmc_sampler = get_sampler(diff_model_gmm, diff_model_bar, diff_proc, config)

    if config.mcmc_method is None:
        prod_samples, trajs = prod_sampler.sample(
            config.num_samples, device, th.Size((2,)), verbose=True, save_traj=config.save_traj
        )
    else:
        prod_samples, trajs = prod_sampler.mcmc_sample(
            config.num_samples, mcmc_sampler, device, th.Size((2,)), verbose=True, save_traj=config.save_traj
        )
    prod_samples = prod_samples.detach().cpu()
    th.save(prod_samples, sim_dir / f"prod_samples.th")
    if config.save_traj:
        print("Saving full traj.")
        # full_trajs is a list of T tensors of shape (B, D, D)
        # th.stack turns the list into a single tensor (T, B, D, D).
        th.save(th.stack(trajs), sim_dir / f"trajs.th")
        if config.mcmc_method is not None:
            if isinstance(mcmc_sampler, MCMCMHCorrSampler):
                mcmc_sampler.save_stats_to_file(sim_dir, ".p")
    print(f"Results written to '{sim_dir}'")


def get_sampler(diff_model_gmm, diff_model_bar, diff_proc, config: SimulationConfig):
    prod_sampler = ProductCompSampler(
        diff_model_gmm, diff_model_bar, diff_proc, use_reverse_step=config.use_rev, param=config.param
    )
    if config.mcmc_method is None:
        return prod_sampler, None

    assert config.mcmc_stepsize is not None
    assert config.mcmc_steps is not None
    assert config.n_trapets is not None
    step_sizes = {int(t): config.mcmc_stepsize for t in range(0, config.num_diff_steps)}
    if config.mcmc_method == "ula":
        if config.param == "score":
            mcmc_sampler = AnnealedULAScoreSampler(config.mcmc_steps, step_sizes, prod_sampler.grad)
        else:
            mcmc_sampler = AnnealedULAEnergySampler(config.mcmc_steps, step_sizes, prod_sampler.grad)

    elif config.mcmc_method == "la":
        if config.param == "score":
            mcmc_sampler = AnnealedLAScoreSampler(
                config.mcmc_steps, step_sizes, prod_sampler.grad, n_trapets=config.n_trapets
            )
        else:
            mcmc_sampler = AnnealedLAEnergySampler(config.mcmc_steps, step_sizes, prod_sampler.grad)
    elif config.mcmc_method == "hmc":
        step_sizes = {int(t): config.mcmc_stepsize for t in range(0, config.num_diff_steps)}
        damping_coeff = 0.5
        leapfrog_steps = 3
        if config.param == "score":
            mcmc_sampler = AnnealedHMCScoreSampler(
                config.mcmc_steps,
                step_sizes,
                damping_coeff,
                th.ones_like(diff_proc.betas),
                leapfrog_steps,
                prod_sampler.grad,
            )
        else:
            mcmc_sampler = AnnealedHMCEnergySampler(
                config.mcmc_steps,
                step_sizes,
                damping_coeff,
                th.ones_like(diff_proc.betas),
                leapfrog_steps,
                prod_sampler.grad,
            )

    elif config.mcmc_method == "uhmc":
        damping_coeff = 0.5
        leapfrog_steps = 3
        if config.param == "score":
            mcmc_sampler = AnnealedUHMCScoreSampler(
                config.mcmc_steps,
                step_sizes,
                damping_coeff,
                th.ones_like(diff_proc.betas),
                leapfrog_steps,
                prod_sampler.grad,
            )
        else:
            mcmc_sampler = AnnealedUHMCEnergySampler(
                config.mcmc_steps,
                step_sizes,
                damping_coeff,
                th.ones_like(diff_proc.betas),
                leapfrog_steps,
                prod_sampler.grad,
            )

    elif config.mcmc_method == "hmc":
        step_sizes = {int(t): config.mcmc_stepsize for t in range(0, config.num_diff_steps)}
        damping_coeff = 0.5
        leapfrog_steps = 3
        if config.param == "score":
            mcmc_sampler = AnnealedHMCScoreSampler(
                config.mcmc_steps,
                step_sizes,
                damping_coeff,
                th.ones_like(diff_proc.betas),
                leapfrog_steps,
                prod_sampler.grad,
            )
        else:
            mcmc_sampler = AnnealedHMCEnergySampler(
                config.mcmc_steps,
                step_sizes,
                damping_coeff,
                th.ones_like(diff_proc.betas),
                leapfrog_steps,
                prod_sampler.grad,
            )
    else:
        raise ValueError(f"Incorrect sampler '{config.mcmc_method}'")

    return prod_sampler, mcmc_sampler


def load_diff_model(diff_model_path, T, device):
    assert diff_model_path.exists(), f"Model '{diff_model_path}' does not exist."
    diff_model = ResnetDiffusionModel(num_diff_steps=T)
    diff_model.load_state_dict(th.load(diff_model_path))
    diff_model.to(device)
    diff_model.eval()
    return diff_model


def parse_args():
    parser = ArgumentParser(prog="Sample from 2D product composition")
    parser.add_argument("--config", type=Path, required=True, help="Config file path")
    return parser.parse_args()


if __name__ == "__main__":
    main()
