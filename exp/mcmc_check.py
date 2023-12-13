"""Simulation script for cluster"""
import sys


sys.path.append(".")
from pathlib import Path
from argparse import ArgumentParser
import torch as th
from src.diffusion.base import DiffusionSampler
from src.diffusion.beta_schedules import (
    improved_beta_schedule,
    linear_beta_schedule,
    respaced_beta_schedule,
)
from src.utils.net import get_device, Device
from src.utils.seeding import set_seed
from src.model.guided_diff.unet import load_pretrained_diff_unet
from src.model.guided_diff.classifier import load_guided_classifier
from src.model.resnet import load_classifier_t
from src.model.unet import load_mnist_diff
from src.guidance.base import GuidanceSampler, MCMCGuidanceSampler
from src.guidance.classifier_full import ClassifierFullGuidance
from src.samplers.mcmc import AnnealedHMCScoreSampler
from src.data.mnist import get_mnist_data_loaders
from exp.utils import SimulationConfig, setup_results_dir, get_step_size


@th.no_grad()
def main():
    seed = 0
    set_seed(seed)
    args = parse_args()
    config = SimulationConfig.from_json(args.config)
    assert config.num_samples % config.batch_size == 0, "num_samples should be a multiple of batch_size"
    # Setup and assign a directory where simulation results are saved.
    sim_dir = setup_results_dir(config)
    device = get_device(Device.GPU)

    models_dir = Path.cwd() / "models"
    diff_model_path = models_dir / f"{config.diff_model}.pt"
    assert diff_model_path.exists(), f"Model '{diff_model_path}' does not exist."
    assert not (config.class_cond and "uncond" in config.diff_model)
    classifier_path = models_dir / f"{config.classifier}.pt"
    assert classifier_path.exists(), f"Model '{classifier_path}' does not exist."

    # Hyper/meta params
    channels, image_size = config.num_channels, config.image_size
    # Load diff. and classifier models
    if "mnist" in config.diff_model:
        channels, image_size = 1, 28
        beta_schedule, post_var = improved_beta_schedule, "beta"
        num_classes = 10
        diff_model = load_mnist_diff(diff_model_path, device)
        diff_model.eval()
        classifier = load_classifier_t(model_path=classifier_path, dev=device)
        classifier.eval()
    elif "256x256_diffusion" in config.diff_model:
        beta_schedule, post_var = linear_beta_schedule, "learned"
        num_classes = 1000
        diff_model = load_pretrained_diff_unet(model_path=diff_model_path, dev=device, class_cond=config.class_cond)
        diff_model.eval()
        classifier = load_guided_classifier(model_path=classifier_path, dev=device, image_size=image_size)
        classifier.eval()
    else:
        print(f"Incorrect model '{args.diff_model}'")

    betas, time_steps = respaced_beta_schedule(
        original_betas=beta_schedule(num_timesteps=config.num_diff_steps),
        T=config.num_diff_steps,
        respaced_T=config.num_respaced_diff_steps,
    )
    # print("betas", betas[0], beta_schedule(1000)[0])
    diff_sampler = DiffusionSampler(betas, time_steps, posterior_variance=post_var)
    guidance = ClassifierFullGuidance(classifier, lambda_=config.guid_scale)

    if config.mcmc_method is None:
        guid_sampler = GuidanceSampler(diff_model, diff_sampler, guidance, diff_cond=config.class_cond)
    else:
        assert config.mcmc_steps is not None and config.mcmc_bounds is not None
        step_sizes = get_step_size(
            models_dir / "step_sizes", config.name, config.num_respaced_diff_steps, config.mcmc_bounds
        )
        mcmc_sampler = AnnealedHMCScoreSampler(config.mcmc_steps, step_sizes, 0.9, diff_sampler.betas, 3, None)
        guid_sampler = MCMCGuidanceSampler(
            diff_model=diff_model,
            diff_proc=diff_sampler,
            guidance=guidance,
            mcmc_sampler=mcmc_sampler,
            reverse=True,
            diff_cond=config.class_cond,
        )
    factor = 100
    step_sizes = {t: s * factor for t, s in step_sizes.items()}
    img, label = get_mnist_batch(1)
    samples = [img.clone()]
    img, label = img.to(device), label.to(device)
    mcmc_sampler.set_gradient_function(guid_sampler.grad)
    x_tau = img.clone()
    num_mcmc_steps = 10
    # for t, s in step_sizes.items():
    #     print(f"{t}: {s}")
    for tau in range(num_mcmc_steps):
        print(f"{tau+1}/{num_mcmc_steps}")
        x_tau_plus_1 = mcmc_sampler.sample_step(x_tau, 0, 0, label)
        samples.append(x_tau_plus_1.clone().detach().cpu())
        x_tau = x_tau_plus_1
    samples = th.stack(samples)
    th.save(samples, Path.cwd() / "results/single_sample_hmc.th")
    print(mcmc_sampler.accept_ratio)


def get_mnist_batch(num_samples):
    train, _ = get_mnist_data_loaders(num_samples)
    batch = next(iter(train))
    labels = batch["label"]
    imgs = batch["pixel_values"]
    return imgs, labels


def parse_args():
    parser = ArgumentParser(prog="Sample from diffusion model")
    parser.add_argument("--config", type=Path, required=True, help="Config file path")
    parser.add_argument(
        "--sim_batch", type=int, default=0, help="Simulation batch index, indexes parallell simulations."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
