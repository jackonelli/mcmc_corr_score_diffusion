"""Script for sampling with classifier-full guidance with MCMC and adaptive step size"""

import sys


sys.path.append(".")
from argparse import ArgumentParser
from pathlib import Path
import pickle
import torch as th
from src.guidance.base import MCMCGuidanceSampler, MCMCGuidanceSamplerStacking
from src.guidance.classifier_full import ClassifierFullGuidance
from src.samplers.mcmc import (
    AnnealedHMCScoreSampler,
    AdaptiveStepSizeMCMCSamplerWrapper,
    AdaptiveStepSizeMCMCSamplerWrapperSmallBatchSize,
    AnnealedLAScoreSampler,
)
from src.model.resnet import load_classifier_t
from src.utils.net import Device, get_device
from src.diffusion.base import DiffusionSampler
from src.diffusion.beta_schedules import (
    improved_beta_schedule,
    respaced_beta_schedule,
    linear_beta_schedule,
)
from src.model.guided_diff.unet import load_guided_diff_unet
from src.model.guided_diff.classifier import load_guided_classifier
from src.model.unet import load_mnist_diff
from exp.utils import timestamp
from src.utils.seeding import set_seed


def main():
    args = parse_args()
    accept_rate_bound_pct = [int(x) for x in args.accept_rate_bound]
    set_seed(args.seed)
    sim_dir = _setup_results_dir(Path.cwd() / "results", args)
    device = get_device(Device.GPU)
    models_dir = Path.cwd() / "models"
    # uncond_diff = load_mnist_diff(models_dir / "uncond_unet_mnist.pt", device)
    # classifier = _load_class(models_dir / "resnet_classifier_t_mnist.pt", device)
    diff_model_path = models_dir / f"{args.diff_model}.pt"
    class_model_path = models_dir / f"{args.class_model}.pt"
    if "mnist" in args.diff_model:
        channels, image_size = 1, 28
        beta_schedule = improved_beta_schedule
        diff_model = load_mnist_diff(diff_model_path, device)
        classifier = load_classifier_t(models_dir / class_model_path, device)
        posterior_variance = "beta"
        num_classes = 10
    elif "256x256_diffusion" in args.diff_model:
        channels, image_size = 3, 256
        beta_schedule = linear_beta_schedule
        diff_model = load_guided_diff_unet(model_path=diff_model_path, dev=device, class_cond=args.class_cond)
        diff_model.eval()
        if args.class_cond:
            print("Using class conditional diffusion model")
        classifier = load_guided_classifier(model_path=class_model_path, dev=device, image_size=image_size)
        classifier.eval()
        posterior_variance = "learned"
        num_classes = 1000

    T = args.num_diff_steps
    respaced_T = args.respaced_num_diff_steps

    betas, time_steps = respaced_beta_schedule(
        original_betas=beta_schedule(num_timesteps=T),
        T=T,
        respaced_T=respaced_T,
    )

    diff_sampler = DiffusionSampler(betas, time_steps, posterior_variance=posterior_variance)
    diff_sampler.to(device)

    # Default values
    a = 1  # 0.05
    b = 1  # 1.6
    step_sizes = {t.item(): a * beta**b for t, beta in zip(time_steps, betas)}
    # step_sizes = a * diff_sampler.betas**b

    batch_size = args.batch_size
    num_samples = args.num_samples

    mcmc_steps = args.n_mcmc_steps
    if args.mcmc == "HMC":
        mcmc_sampler = AnnealedHMCScoreSampler(mcmc_steps, step_sizes, 0.9, diff_sampler.betas, 3, None)
    else:
        mcmc_sampler = AnnealedLAScoreSampler(mcmc_steps, step_sizes, None)
    max_iter = args.max_iter

    guidance = ClassifierFullGuidance(classifier, lambda_=args.guid_scale)
    th.manual_seed(0)
    classes = th.randint(num_classes, (num_samples,), dtype=th.int64)
    if batch_size < num_samples:
        sampler = AdaptiveStepSizeMCMCSamplerWrapperSmallBatchSize(
            sampler=mcmc_sampler,
            accept_rate_bound=[a / 100 for a in accept_rate_bound_pct],
            time_steps=time_steps,
            batch_size=batch_size,
            device=device,
            max_iter=max_iter,
        )

        guided_sampler = MCMCGuidanceSamplerStacking(
            diff_model=diff_model,
            diff_proc=diff_sampler,
            guidance=guidance,
            mcmc_sampler=sampler,
            diff_cond=args.class_cond,
        )
        samples, _ = guided_sampler.sample_stacking(
            num_samples, batch_size, classes, device, th.Size((channels, image_size, image_size))
        )
    else:
        print("Running Adaptive MCMC sampler")
        sampler = AdaptiveStepSizeMCMCSamplerWrapper(
            sampler=mcmc_sampler,
            accept_rate_bound=[a / 100 for a in accept_rate_bound_pct],
            time_steps=time_steps,
            max_iter=max_iter,
        )
        guided_sampler = MCMCGuidanceSampler(
            diff_model=diff_model,
            diff_proc=diff_sampler,
            guidance=guidance,
            mcmc_sampler=sampler,
            diff_cond=args.class_cond,
        )
        samples, _ = guided_sampler.sample(num_samples, classes, device, th.Size((channels, image_size, image_size)))

    adaptive_step_sizes = sampler.res
    lower_bound, upper_bound = accept_rate_bound_pct
    save_path = sim_dir / f"mnist_{respaced_T}_{lower_bound}_{upper_bound}.p"
    pickle.dump(adaptive_step_sizes, open(save_path, "wb"))


import json


def _setup_results_dir(res_dir: Path, args) -> Path:
    assert res_dir.exists()
    sim_dir = res_dir / f"find_stepsize_{timestamp()}"
    sim_dir.mkdir(exist_ok=True)
    args_dict = vars(args)
    with open(sim_dir / "args.json", "w") as file:
        json.dump(args_dict, file, indent=2)

    return sim_dir


def parse_args():
    parser = ArgumentParser(prog="Find step size for MCMC for classifier-full guidance")
    parser.add_argument("--guid_scale", default=1.0, type=float, help="Guidance scale")
    parser.add_argument("--num_diff_steps", default=1000, type=int, help="Num diffusion steps")
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
    parser.add_argument("--num_samples", default=120, type=int, help="Number of samples for estimate acceptance ratio")
    parser.add_argument(
        "--accept_rate_bound", default=[0.6, 0.8], nargs="+", type=float, help="Acceptance ratio bounds"
    )
    parser.add_argument("--max_iter", default=20, type=int, help="Number of search iterations per time step")
    parser.add_argument("--mcmc", default="HMC", type=str, choices=["HMC", "LA"], help="Type of MCMC sampler")
    parser.add_argument("--n_mcmc_steps", default=1, type=int, help="Number of MCMC steps")
    parser.add_argument(
        "--respaced_num_diff_steps",
        default=100,
        type=int,
        help="Number of respaced diffusion steps (fewer than or equal to num_diff_steps)",
    )
    parser.add_argument("--diff_model", type=str, help="Diffusion model file (withouth '.pt' extension)")
    parser.add_argument("--class_model", type=str, help="Classifier model file (withouth '.pt' extension)")
    parser.add_argument("--class_cond", action="store_true", help="Use classconditional diff. model")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
