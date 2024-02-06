"""Sampled from diffusion models

Unguided diffusion sampling for baselines
"""
import sys


sys.path.append(".")
from pathlib import Path
from argparse import ArgumentParser
import torch as th

# Sampling
from src.diffusion.base import DiffusionSampler
from src.diffusion.beta_schedules import (
    improved_beta_schedule,
    linear_beta_schedule,
    respaced_beta_schedule,
)

# Diff models
from src.model.cifar.unet import load_model as load_unet_diff_model
from src.model.cifar.unet_ho import load_model as load_unet_ho_diff_model
from src.model.cifar.unet_drop import load_model as load_unet_drop_diff_model
from src.model.guided_diff.unet import load_pretrained_diff_unet
from src.model.unet import load_mnist_diff

# Exp setup
from src.utils.seeding import set_seed
from src.data.cifar import CIFAR_100_NUM_CLASSES, CIFAR_IMAGE_SIZE, CIFAR_NUM_CHANNELS
from src.utils.net import get_device, Device
from exp.utils import UnguidedSimulationConfig, setup_results_dir, get_step_size


def main():
    args = parse_args()
    config = UnguidedSimulationConfig.from_json(args.config)
    assert config.num_samples % config.batch_size == 0, "num_samples should be a multiple of batch_size"
    set_seed(config.seed)

    # Setup and assign a directory where simulation results are saved.
    sim_dir = setup_results_dir(config, args.job_id)
    device = get_device(Device.GPU)

    # Load diff. and classifier models
    (diff_model, dataset, beta_schedule, post_var, energy_param) = load_models(config, device, config.num_diff_steps)
    dataset_name, image_size, num_classes, num_channels = dataset

    betas, time_steps = respaced_beta_schedule(
        original_betas=beta_schedule(num_timesteps=config.num_diff_steps),
        T=config.num_diff_steps,
        respaced_T=config.num_respaced_diff_steps,
    )
    diff_sampler = DiffusionSampler(betas, time_steps, posterior_variance=post_var, energy=energy_param)

    print(f"Sampling with {'energy' if energy_param else 'score'} parameterisation")
    for batch in range(config.num_samples // config.batch_size):
        print(f"{(batch+1) * config.batch_size}/{config.num_samples}")
        classes = th.randint(low=0, high=num_classes, size=(config.batch_size,)).long().to(device)
        samples, _ = diff_sampler.sample(
            diff_model,
            config.batch_size,
            device,
            th.Size((num_channels, image_size, image_size)),
            verbose=True,
        )
        samples = samples.detach().cpu()
        th.save(samples, sim_dir / f"samples_{args.sim_batch}_{batch}.th")
        th.save(classes, sim_dir / f"classes_{args.sim_batch}_{batch}.th")
    print(f"Results written to '{sim_dir}'")


def _get_beta_schedule(name):
    if 'lin' in name:
        beta_schedule = linear_beta_schedule
    elif 'cos' in name:
        beta_schedule = improved_beta_schedule
    else:
        raise ValueError('Invalid beta schedule')
    return beta_schedule, "beta"


def _get_model(name, diff_model_path, device, energy_param, image_size, num_steps):
    if "small" in name:
        diff_model = load_unet_diff_model(
            diff_model_path, device, image_size=image_size, energy_param=energy_param
        )
    elif "large2" in name:
        diff_model = load_unet_drop_diff_model(
            diff_model_path,
            device,
            energy_param=energy_param,
            T = num_steps
        )
    elif "large" in name:
        diff_model = load_unet_ho_diff_model(
            diff_model_path, device, energy_param=energy_param
        )
    else:
        raise ValueError("Not specified model size")
    return diff_model


def load_models(config, device, num_steps):
    diff_model_name = f"{config.diff_model}"
    diff_model_path = MODELS_DIR / f"{diff_model_name}"
    assert diff_model_path.exists(), f"Model '{diff_model_path}' does not exist."
    energy_param = "energy" in diff_model_name

    assert not (config.class_cond and "uncond" in config.diff_model)
    if "mnist" in diff_model_name:
        dataset_name = "mnist"
        beta_schedule, post_var = improved_beta_schedule, "beta"
        image_size, num_classes, num_channels = (28, 10, 1)
        diff_model = load_mnist_diff(diff_model_path, device)
        diff_model.eval()
    elif "cifar100" in diff_model_name:
        dataset_name = "cifar100"
        beta_schedule, post_var = _get_beta_schedule(diff_model_name)
        image_size, num_classes, num_channels = (CIFAR_IMAGE_SIZE, CIFAR_100_NUM_CLASSES, CIFAR_NUM_CHANNELS)
        diff_model = _get_model(diff_model_name, diff_model_path, device, energy_param, CIFAR_IMAGE_SIZE)
        diff_model.eval()
    elif "cifar10" in diff_model_name:
        dataset_name = "cifar10"
        beta_schedule, post_var = _get_beta_schedule(diff_model_name)
        image_size, num_classes, num_channels = (CIFAR_IMAGE_SIZE, 10, CIFAR_NUM_CHANNELS)
        diff_model = _get_model(diff_model_name, diff_model_path, device, energy_param, CIFAR_IMAGE_SIZE, num_steps)
        diff_model.eval()
    elif f"{config.image_size}x{config.image_size}_diffusion" in diff_model_name:
        dataset_name = "imagenet"
        beta_schedule, post_var = linear_beta_schedule, "learned"
        image_size, num_classes, num_channels = (config.image_size, 1000, 3)
        diff_model = load_pretrained_diff_unet(
            model_path=diff_model_path, dev=device, class_cond=config.class_cond, image_size=image_size
        )
        diff_model.eval()
    else:
        print(f"Incorrect model '{diff_model_name}'")
        raise ValueError
    return (
        diff_model,
        (dataset_name, image_size, num_classes, num_channels),
        beta_schedule,
        post_var,
        energy_param,
    )


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
    return parser.parse_args()


if __name__ == "__main__":
    main()
