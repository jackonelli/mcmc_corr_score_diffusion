"""Simulation script for cluster"""
import sys


sys.path.append(".")
from pathlib import Path
from argparse import ArgumentParser
import torch as th
from src.diffusion.base import DiffusionSampler
from src.diffusion.beta_schedules import (
    linear_beta_schedule,
    respaced_beta_schedule,
)
from src.utils.net import get_device, Device
from src.model.guided_diff.unet import NUM_CLASSES, load_guided_diff_unet
from src.model.guided_diff.classifier import load_guided_classifier
from src.guidance.base import GuidanceSampler, MCMCGuidanceSampler
from src.guidance.classifier_full import ClassifierFullGuidance
from src.samplers.mcmc import AnnealedHMCScoreSampler
from exp.utils import SimulationConfig, setup_results_dir, get_step_size


def main():
    args = parse_args()
    config = SimulationConfig.from_json(args.config)
    assert config.num_samples % config.batch_size == 0, "num_samples should be a multiple of batch_size"
    # Setup and assign a directory where simulation results are saved.
    sim_dir = setup_results_dir(config)
    device = get_device(Device.GPU)

    # Hyper/meta params
    channels, image_size = config.num_channels, config.image_size
    beta_schedule, post_var = linear_beta_schedule, "learned"
    betas, time_steps = respaced_beta_schedule(
        original_betas=beta_schedule(num_timesteps=config.num_diff_steps),
        T=config.num_diff_steps,
        respaced_T=config.num_respaced_diff_steps,
    )

    # Load diff. model
    models_dir = Path.cwd() / "models"
    diff_model_path = models_dir / f"{config.diff_model}.pt"
    assert diff_model_path.exists(), f"Model '{diff_model_path}' does not exist."
    assert not (config.class_cond and "uncond" in config.diff_model)
    diff_model = load_guided_diff_unet(model_path=diff_model_path, dev=device, class_cond=config.class_cond)
    diff_model.eval()

    # Load classifier
    classifier_path = models_dir / f"{config.classifier}.pt"
    assert classifier_path.exists(), f"Model '{classifier_path}' does not exist."
    classifier = load_guided_classifier(model_path=classifier_path, dev=device, image_size=image_size)
    classifier.eval()

    diff_sampler = DiffusionSampler(betas, time_steps, posterior_variance=post_var)
    guidance = ClassifierFullGuidance(classifier, lambda_=config.guid_scale)

    if config.mcmc_method is None:
        guid_sampler = GuidanceSampler(diff_model, diff_sampler, guidance, diff_cond=config.class_cond)
    else:
        step_sizes = get_step_size(models_dir / "step_sizes", config.mcmc_bounds)
        mcmc_sampler = AnnealedHMCScoreSampler(config.mcmc_steps, step_sizes, 0.9, diff_sampler.betas, 3, None)
        guid_sampler = MCMCGuidanceSampler(
            diff_model=diff_model,
            diff_proc=diff_sampler,
            guidance=guidance,
            mcmc_sampler=mcmc_sampler,
            reverse=True,
            diff_cond=config.class_cond,
        )

    print("Sampling...")
    for batch in range(config.num_samples // config.batch_size):
        print(f"{(batch+1) * config.batch_size}/{config.num_samples}")
        classes = th.randint(low=0, high=NUM_CLASSES, size=(config.batch_size,)).long().to(device)
        samples, _ = guid_sampler.sample(
            config.batch_size, classes, device, th.Size((channels, image_size, image_size)), verbose=True
        )
        samples = samples.detach().cpu()
        th.save(samples, sim_dir / f"samples_{args.sim_batch}_{batch}.th")
        th.save(classes, sim_dir / f"classes_{args.sim_batch}_{batch}.th")


def parse_args():
    parser = ArgumentParser(prog="Sample from diffusion model")
    parser.add_argument("--config", type=Path, required=True, help="Config file path")
    parser.add_argument(
        "--sim_batch", type=int, default=0, help="Simulation batch index, indexes parallell simulations."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
