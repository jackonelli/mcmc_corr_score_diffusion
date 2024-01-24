"""Simulation script for cluster"""
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
from src.guidance.base import GuidanceSampler, MCMCGuidanceSampler
from src.guidance.classifier_full import ClassifierFullGuidance
from src.samplers.mcmc import (
    AnnealedHMCScoreSampler,
    AnnealedLAScoreSampler,
    AnnealedUHMCScoreSampler,
    AnnealedULAScoreSampler,
)

# Diff models
from src.model.cifar.unet import load_model as load_unet_diff_model
from src.model.guided_diff.unet import load_pretrained_diff_unet
from src.model.unet import load_mnist_diff

# Classifiers
from src.model.cifar.class_t import load_classifier_t as load_unet_classifier_t
from src.model.resnet import load_classifier_t as load_resnet_classifier_t
from src.model.guided_diff.classifier import load_guided_classifier as load_guided_diff_classifier_t

# Exp setup
from src.utils.seeding import set_seed
from src.data.cifar import CIFAR_100_NUM_CLASSES, CIFAR_IMAGE_SIZE, CIFAR_NUM_CHANNELS
from src.utils.net import get_device, Device
from exp.utils import SimulationConfig, setup_results_dir, get_step_size


def main():
    args = parse_args()
    config = SimulationConfig.from_json(args.config)
    assert config.num_samples % config.batch_size == 0, "num_samples should be a multiple of batch_size"
    set_seed(config.seed)

    # Setup and assign a directory where simulation results are saved.
    sim_dir = setup_results_dir(config, args.job_id)
    device = get_device(Device.GPU)

    # Load diff. and classifier models
    (
        diff_model,
        classifier,
        dataset,
        beta_schedule,
        post_var,
    ) = load_models(config, device)
    dataset_name, image_size, num_classes, num_channels = dataset

    betas, time_steps = respaced_beta_schedule(
        original_betas=beta_schedule(num_timesteps=config.num_diff_steps),
        T=config.num_diff_steps,
        respaced_T=config.num_respaced_diff_steps,
    )
    diff_sampler = DiffusionSampler(betas, time_steps, posterior_variance=post_var)
    guidance = ClassifierFullGuidance(classifier, lambda_=config.guid_scale)
    guid_sampler = get_guid_sampler(config, diff_model, diff_sampler, guidance, time_steps, dataset_name)

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
    print(f"Results written to '{sim_dir}'")


def load_models(config, device):
    diff_model_name = f"{config.diff_model}"
    diff_model_path = MODELS_DIR / f"{diff_model_name}"
    assert diff_model_path.exists(), f"Model '{diff_model_path}' does not exist."

    assert not (config.class_cond and "uncond" in config.diff_model)
    classifier_name = f"{config.classifier}"
    classifier_path = MODELS_DIR / f"{classifier_name}"
    assert classifier_path.exists(), f"Model '{classifier_path}' does not exist."
    if "mnist" in diff_model_name:
        dataset_name = "mnist"
        beta_schedule, post_var = improved_beta_schedule, "beta"
        image_size, num_classes, num_channels = (28, 10, 1)
        diff_model = load_mnist_diff(diff_model_path, device)
        diff_model.eval()
        classifier = load_resnet_classifier_t(
            model_path=classifier_path,
            dev=device,
            num_channels=num_channels,
            num_classes=num_classes,
        )
        classifier.eval()
    elif "cifar100" in diff_model_name:
        dataset_name = "cifar100"
        beta_schedule, post_var = improved_beta_schedule, "beta"
        image_size, num_classes, num_channels = (CIFAR_IMAGE_SIZE, CIFAR_100_NUM_CLASSES, CIFAR_NUM_CHANNELS)
        diff_model = load_unet_diff_model(diff_model_path, device, image_size=CIFAR_IMAGE_SIZE)
        diff_model.eval()
        classifier = select_cifar_classifier(model_path=classifier_path, dev=device)
        classifier.eval()
    elif f"{config.image_size}x{config.image_size}_diffusion" in diff_model_name:
        dataset_name = "imagenet"
        beta_schedule, post_var = linear_beta_schedule, "learned"
        image_size, num_classes, num_channels = (config.image_size, 1000, 3)
        diff_model = load_pretrained_diff_unet(
            model_path=diff_model_path, dev=device, class_cond=config.class_cond, image_size=image_size
        )
        diff_model.eval()
        classifier = load_guided_diff_classifier_t(model_path=classifier_path, dev=device, image_size=image_size)
        classifier.eval()
    else:
        print(f"Incorrect model '{diff_model_name}'")
        raise ValueError
    return diff_model, classifier, (dataset_name, image_size, num_classes, num_channels), beta_schedule, post_var


def select_cifar_classifier(model_path: Path, dev):
    arch = parse_arch(model_path)
    if arch == "unet":
        class_t = load_unet_classifier_t(None, dev)
    elif arch == "resnet":
        class_t = load_resnet_classifier_t(
            model_path=model_path,
            dev=dev,
            emb_dim=112,
            num_classes=CIFAR_100_NUM_CLASSES,
            num_channels=CIFAR_NUM_CHANNELS,
        ).to(dev)
    elif arch == "guided_diff":
        class_t = load_guided_diff_classifier_t(
            model_path=None, dev=dev, image_size=CIFAR_IMAGE_SIZE, num_classes=CIFAR_100_NUM_CLASSES
        ).to(dev)
    else:
        raise ValueError(f"Incorrect model arch: {arch}")
    return class_t


def parse_arch(model_path: Path):
    """Get model architecture from model name

    Models are (supposed) to be stored as <dataset>_<arch>_class_t.{ckpt,th}
    """
    name = model_path.name
    available_archs = ["unet", "resnet", "guided_diff"]
    return next(filter(lambda arch: arch in name, available_archs))


def get_guid_sampler(config, diff_model, diff_sampler, guidance, time_steps, dataset_name):
    if config.mcmc_method is None:
        guid_sampler = GuidanceSampler(diff_model, diff_sampler, guidance, diff_cond=config.class_cond)
    else:
        assert config.mcmc_steps is not None
        assert config.mcmc_method is not None
        assert config.mcmc_stepsizes is not None
        if config.mcmc_stepsizes["load"]:
            print("Load step sizes for MCMC.")
            step_sizes = get_step_size(
                MODELS_DIR / "step_sizes", dataset_name, config.mcmc_method, config.mcmc_stepsizes["bounds"]
            )
        else:
            print("Use parameterized step sizes for MCMC.")
            if config.mcmc_stepsizes["beta_schedule"] == "lin":
                beta_schedule_mcmc = linear_beta_schedule
            elif config.mcmc_stepsizes["beta_schedule"] == "cos":
                beta_schedule_mcmc = improved_beta_schedule
            else:
                print("mcmc_stepsizes.beta_schedule must be 'lin' or 'cos'.")
                raise ValueError
            betas_mcmc, _ = respaced_beta_schedule(
                original_betas=beta_schedule_mcmc(num_timesteps=config.num_diff_steps),
                T=config.num_diff_steps,
                respaced_T=config.num_respaced_diff_steps,
            )
            a = config.mcmc_stepsizes["params"]["factor"]
            b = config.mcmc_stepsizes["params"]["exponent"]
            step_sizes = {int(t.item()): a * beta**b for (t, beta) in zip(time_steps, betas_mcmc)}

        if config.mcmc_method == "hmc":
            mcmc_sampler = AnnealedHMCScoreSampler(config.mcmc_steps, step_sizes, 0.9, diff_sampler.betas, 3, None)
        elif config.mcmc_method == "la":
            assert config.n_trapets is not None
            mcmc_sampler = AnnealedLAScoreSampler(config.mcmc_steps, step_sizes, None, n_trapets=config.n_trapets)
        elif config.mcmc_method == "uhmc":
            mcmc_sampler = AnnealedUHMCScoreSampler(config.mcmc_steps, step_sizes, 0.9, diff_sampler.betas, 3, None)
        elif config.mcmc_method == "ula":
            mcmc_sampler = AnnealedULAScoreSampler(config.mcmc_steps, step_sizes, None)
        else:
            raise ValueError(f"Incorrect MCMC method: '{config.mcmc_method}'")

        guid_sampler = MCMCGuidanceSampler(
            diff_model=diff_model,
            diff_proc=diff_sampler,
            guidance=guidance,
            mcmc_sampler=mcmc_sampler,
            reverse=True,
            diff_cond=config.class_cond,
        )
    return guid_sampler


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
