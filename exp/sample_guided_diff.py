"""Sampled from guided diffusion models"""
import sys


sys.path.append(".")
from pathlib import Path
from argparse import ArgumentParser
import torch as th

# Sampling
from src.diffusion.base import DiffusionSampler
from src.diffusion.beta_schedules import (
    respaced_beta_schedule,
)
from src.guidance.classifier_full import ClassifierFullGuidance
from src.samplers.utils import get_guid_sampler

from src.model.utils import load_models

# Exp setup
from src.utils.seeding import set_seed
from src.utils.net import get_device, Device
from exp.utils import SimulationConfig, setup_results_dir


def main():
    args = parse_args()
    config = SimulationConfig.from_json(args.config)
    assert config.num_samples % config.batch_size == 0, "num_samples should be a multiple of batch_size"
    set_seed(config.seed)

    # Setup and assign a directory where simulation results are saved.
    sim_dir = setup_results_dir(config, args.job_id)
    device = get_device(Device.GPU)

    # Load diff. and classifier models
    (diff_model, classifier, dataset, beta_schedule, post_var, energy_param) = load_models(config, device, MODELS_DIR)
    dataset_name, image_size, num_classes, num_channels = dataset

    betas, time_steps = respaced_beta_schedule(
        original_betas=beta_schedule(num_timesteps=config.num_diff_steps),
        T=config.num_diff_steps,
        respaced_T=config.num_respaced_diff_steps,
    )
    diff_sampler = DiffusionSampler(betas, time_steps, posterior_variance=post_var)
    guidance = ClassifierFullGuidance(classifier, lambda_=config.guid_scale)
    guid_sampler = get_guid_sampler(config, diff_model, diff_sampler, guidance, time_steps, dataset_name, energy_param,
                                    MODELS_DIR, save_grad=args.save_grad)

    print("Sampling...")
    for batch in range(config.num_samples // config.batch_size):
        print(f"{(batch+1) * config.batch_size}/{config.num_samples}")
        classes = th.randint(low=0, high=num_classes, size=(config.batch_size,)).long().to(device)
        samples, _ = guid_sampler.sample(
            config.batch_size, classes, device, th.Size((num_channels, image_size, image_size)), verbose=True
        )
        samples = samples.detach().cpu()
        th.save(samples, sim_dir / f"samples_{args.sim_batch}_{batch}.th")
        th.save(classes, sim_dir / f"classes_{args.sim_batch}_{batch}.th")
        if (config.mcmc_method == "hmc" or config.mcmc_method == "la") and args.sim_batch == 1 and batch == 0:
            guid_sampler.mcmc_sampler.save_stats_to_file(dir_=sim_dir, suffix=f"{args.sim_batch}_{batch}")
        if args.save_grad and args.sim_batch == 1 and batch == 0:
            guid_sampler.save_grads_to_file(dir_=sim_dir, suffix=f"{args.sim_batch}_{batch}")
    print(f"Results written to '{sim_dir}'")


MODELS_DIR = Path.cwd() / "models"


def parse_args():
    parser = ArgumentParser(prog="Sample from diffusion model")
    parser.add_argument("--config", type=Path, required=True, help="Config file path")
    parser.add_argument(
        "--job_id", type=int, default=None, help="Simulation batch index, indexes parallell simulations."
    )
    parser.add_argument(
        "--sim_batch", type=int, default=0, help="Simulation batch index, indexes parallell simulations."
    )
    parser.add_argument("--save_grad", action="store_true", help="Save norm of gradients")
    return parser.parse_args()


if __name__ == "__main__":
    main()
