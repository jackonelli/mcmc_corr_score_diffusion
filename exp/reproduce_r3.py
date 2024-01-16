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
from src.model.guided_diff.unet import load_pretrained_diff_unet
from src.model.guided_diff.classifier import load_guided_classifier
from src.guidance.base import GuidanceSampler, MCMCGuidanceSampler
from src.guidance.classifier_full import ClassifierFullGuidance
from src.samplers.mcmc import AnnealedULAScoreSampler, AnnealedUHMCScoreSampler
from exp.utils import SimulationConfig, setup_results_dir
from src.utils.seeding import set_seed


def main():
    args = parse_args()
    config = SimulationConfig.from_json(args.config)
    assert config.num_samples % config.batch_size == 0, "num_samples should be a multiple of batch_size"
    set_seed(config.seed)

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
    num_classes = 1000
    post_var = "learned"
    beta_schedule = linear_beta_schedule
    # beta_schedule = improved_beta_schedule
    diff_model = load_pretrained_diff_unet(model_path=diff_model_path, dev=device, class_cond=config.class_cond)
    diff_model.eval()
    classifier = load_guided_classifier(model_path=classifier_path, dev=device, image_size=image_size)
    classifier.eval()

    betas, time_steps = respaced_beta_schedule(
        original_betas=beta_schedule(num_timesteps=config.num_diff_steps),
        T=config.num_diff_steps,
        respaced_T=config.num_respaced_diff_steps,
    )
    diff_sampler = DiffusionSampler(betas, time_steps, posterior_variance=post_var)
    guidance = ClassifierFullGuidance(classifier, lambda_=config.guid_scale)

    # Gradient function is None here, but is set later in MCMCGuidanceSampler
    if config.mcmc_method is None:
        guid_sampler = GuidanceSampler(diff_model, diff_sampler, guidance, diff_cond=config.class_cond)
    else:
        assert config.mcmc_steps is not None and config.mcmc_method is not None
        if config.mcmc_stepsizes["load"]:
            print("Load step sizes for MCMC.")
            step_sizes = get_step_size(
                models_dir / "step_sizes", dataset_name, config.mcmc_method, config.mcmc_stepsizes["bounds"]
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
            print(f"Using the step size {a} * beta_t^{b}")

        if config.mcmc_method == "uhmc":
            mcmc_sampler = AnnealedUHMCScoreSampler(config.mcmc_steps, step_sizes, 0.9, diff_sampler.betas, 3, None)
        elif config.mcmc_method == "ula":
            mcmc_sampler = AnnealedULAScoreSampler(config.mcmc_steps, step_sizes, None)
        else:
            print(f"Incorrect MCMC method: '{config.mcmc_method}'")

        mcmc_lower_t = config.mcmc_lower_t if config.mcmc_lower_t is not None else 0
        guid_sampler = MCMCGuidanceSampler(
            diff_model=diff_model,
            diff_proc=diff_sampler,
            guidance=guidance,
            mcmc_sampler=mcmc_sampler,
            mcmc_sampling_predicate=lambda t: t > mcmc_lower_t,
            reverse=True,
            diff_cond=config.class_cond,
        )

    print("Sampling...")
    for batch in range(config.num_samples // config.batch_size):
        print(f"{(batch+1) * config.batch_size}/{config.num_samples}")
        classes = th.randint(low=0, high=num_classes, size=(config.batch_size,)).long().to(device)
        samples, full_trajs = guid_sampler.sample(
            config.batch_size,
            classes,
            device,
            th.Size((channels, image_size, image_size)),
            verbose=True,
            save_traj=config.save_traj,
        )
        samples = samples.detach().cpu()
        th.save(samples, sim_dir / f"samples_{args.sim_batch}_{batch}.th")
        th.save(classes, sim_dir / f"classes_{args.sim_batch}_{batch}.th")
        if args.save_traj:
            print("Saving full traj.")
            # full_trajs is a list of T tensors of shape (B, D, D)
            # th.stack turns the list into a single tensor (T, B, D, D).
            th.save(th.stack(full_trajs), sim_dir / f"trajs_{args.sim_batch}_{batch}.th")
    print(f"Results written to '{sim_dir}'")


def parse_args():
    parser = ArgumentParser(prog="Sample from diffusion model")
    parser.add_argument("--config", type=Path, required=True, help="Config file path")
    parser.add_argument("--save_traj", action="store_true", help="Save full trajectories")
    parser.add_argument(
        "--sim_batch", type=int, default=0, help="Simulation batch index, indexes parallell simulations."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
