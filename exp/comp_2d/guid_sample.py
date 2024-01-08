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
from src.model.comp_2d.classifier import load_classifier
from src.guidance.classifier_full import ClassifierFullGuidance
from src.guidance.base import GuidanceSampler, MCMCGuidanceSampler
from src.samplers.mcmc import (
    AnnealedHMCScoreSampler,
    AnnealedLAScoreSampler,
    AnnealedUHMCScoreSampler,
    AnnealedULAScoreSampler,
)
from src.utils.seeding import set_seed


def main():
    T = 100
    num_classes = 8
    guid_scale = 1.0
    x_dim = 2
    args = parse_args()
    set_seed(args.seed)

    # Setup and assign a directory where simulation results are saved.
    sim_dir = Path.cwd() / "results/comp_2d"
    sim_dir.mkdir(exist_ok=True)
    device = get_device(Device.GPU)

    models_dir = Path.cwd() / "models/comp_2d"
    diff_model = load_diff_model(models_dir / "gmm.pt", T, device)
    classifier = load_classifier(models_dir / "class_t_gmm.pt", num_classes, device, num_diff_steps=T)

    betas, time_steps = respaced_beta_schedule(
        original_betas=improved_beta_schedule(num_timesteps=T),
        T=T,
        respaced_T=T,
    )
    diff_proc = DiffusionSampler(betas, time_steps, posterior_variance="beta")
    guidance = ClassifierFullGuidance(classifier, lambda_=guid_scale)

    classes = th.randint(low=0, high=num_classes, size=(args.num_samples,)).long().to(device)

    print("Simple guidance sampling")
    guid_sampler = GuidanceSampler(diff_model, diff_proc, guidance, diff_cond=False)
    samples, _ = guid_sampler.sample(args.num_samples, classes, device, th.Size((x_dim,)), verbose=True)
    samples = samples.detach().cpu()
    classes = classes.detach().cpu()
    th.save(samples, sim_dir / f"guid_samples_gmm.th")
    th.save(classes, sim_dir / f"guid_classes_gmm.th")

    print("HMC guidance sampling")
    mcmc_steps = 10
    step_sizes = {int(t): 0.03 for t in range(0, T)}
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
    classes = classes.detach().cpu()
    th.save(samples, sim_dir / f"hmc_guid_samples_gmm.th")
    th.save(classes, sim_dir / f"hmc_guid_classes_gmm.th")
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
