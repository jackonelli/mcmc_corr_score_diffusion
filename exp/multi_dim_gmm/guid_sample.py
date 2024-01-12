"""Classifier-full guidance sampling from multidimensional GMMs"""
import sys


sys.path.append(".")
import pickle
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
    MCMCMHCorrSampler,
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
        sub_name = f"full_rank_mean_sc_{args.mean_scale}"
    else:
        sub_name = f"rank_{args.low_rank_dim}_{args.mean_scale}"

    sim_dir = Path.cwd() / f"results/multi_dim_gmm_T_{num_diff_steps}/{sub_name}"
    sim_dir.mkdir(exist_ok=True, parents=True)
    device = get_device(Device.GPU)

    models_dir = Path.cwd() / f"models/multi_dim_gmm_T_{num_diff_steps}/{sub_name}"

    betas = improved_beta_schedule(num_timesteps=num_diff_steps)
    time_steps = th.tensor([i for i in range(num_diff_steps)])
    diff_proc = DiffusionSampler(betas, time_steps, posterior_variance="beta")

    x_dim = args.x_dim
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
    if isinstance(guid_sampler.mcmc_sampler, MCMCMHCorrSampler):
        with open(sim_dir / "alpha.p", "wb") as ff:
            pickle.dump(guid_sampler.mcmc_sampler.alpha, ff)


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
    parser.add_argument("--x_dim", type=int, default=10, help="x dim")
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
