"""Simulation script for cluster"""
import sys


sys.path.append(".")
from functools import partial
from pathlib import Path
from argparse import ArgumentParser
import torch as th
import matplotlib.pyplot as plt
from src.diffusion.base import DiffusionSampler
from src.diffusion.beta_schedules import (
    improved_beta_schedule,
    linear_beta_schedule,
    respaced_beta_schedule,
)
from src.model.unet import load_mnist_diff
from src.model.imagenet import load_imagenet_diff
from src.utils.net import get_device, Device
from src.model.guided_diff.unet import load_guided_diff_unet
from exp.utils import SimulationConfig, timestamp


def main():
    args = parse_args()
    config = SimulationConfig.from_json(args.config)
    device = get_device(Device.GPU)
    models_dir = Path.cwd() / "models"
    model_path = models_dir / f"{config.diff_model}.pt"
    assert model_path.exists(), f"Model '{model_path}' does not exist."

    assert not (config.class_cond and "uncond" in config.diff_model)
    diff_model = load_guided_diff_unet(model_path=model_path, dev=device, class_cond=args.class_cond)
    diff_model.eval()

    channels, image_size = config.num_channels, config.image_size
    beta_schedule, var_mode = linear_beta_schedule, "learned"
    betas, time_steps = respaced_beta_schedule(
        original_betas=beta_schedule(num_timesteps=args.num_diff_steps),
        T=args.num_diff_steps,
        respaced_T=args.respaced_num_diff_steps,
    )

    classes = th.ones((args.num_samples,)).long().to(device)
    diff_sampler = DiffusionSampler(betas, time_steps, posterior_variance=var_mode)

    print("Sampling...")
    samples, _ = diff_sampler.sample(
        diff_model, args.num_samples, device, (channels, image_size, image_size), verbose=True
    )
    samples = samples.detach().cpu()
    th.save(samples, Path.cwd() / "outputs" / f"uncond_samples_{args.model}.th")
    if args.plot:
        x = samples[0].permute(1, 2, 0)
        plt.imshow(x)
        plt.show()


def _setup_results_dir(config: SimulationConfig) -> Path:
    assert config.results_dir.exists()
    sim_dir = config.results_dir / f"{config.name}_{timestamp()}"
    sim_dir.mkdir()
    config.save(sim_dir / "config.json")
    return sim_dir


def parse_args():
    parser = ArgumentParser(prog="Sample from diffusion model")
    parser.add_argument("--config", type=Path, help="Config file path")
    parser.add_argument("--sim_batch", type=int, help="Simulation batch index, indexes parallell simulations.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
