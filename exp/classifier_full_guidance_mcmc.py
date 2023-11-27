"""Script for sampling with classifier-full guidance with MCMC"""
import sys

sys.path.append(".")
import pickle
from argparse import ArgumentParser
from pathlib import Path
from functools import partial
import torch as th
from src.guidance.base import MCMCGuidanceSampler
from src.guidance.classifier_full import ClassifierFullGuidance
from src.samplers.mcmc import AnnealedHMCScoreSampler
from src.model.resnet import load_classifier
from src.utils.net import Device, get_device
from src.diffusion.base import DiffusionSampler
from src.diffusion.beta_schedules import improved_beta_schedule, linear_beta_schedule, respaced_beta_schedule
from src.model.unet import load_mnist_diff
from src.utils.vis import plot_samples_grid
from src.model.guided_diff.unet import load_guided_diff_unet
from src.model.guided_diff.classifier import load_guided_classifier


def main():
    args = parse_args()
    device = get_device(Device.GPU)
    models_dir = Path.cwd() / "models"
    diff_model_path = models_dir / f"{args.diff_model}.pt"
    class_model_path = models_dir / f"{args.class_model}.pt"
    num_samples = args.num_samples
    classes = th.ones((num_samples,), dtype=th.int64).to(device)
    T = args.num_diff_steps
    if "mnist" in args.diff_model:
        channels, image_size = 1, 28
        beta_schedule, var_mode = improved_beta_schedule, "beta"
        diff_model = load_mnist_diff(diff_model_path, device)
        classifier = _load_class(models_dir / class_model_path, device)
    elif "256x256_diffusion" in args.diff_model:
        channels, image_size = 3, 256
        beta_schedule, var_mode = linear_beta_schedule, "learned"

        diff_model_proto = load_guided_diff_unet(model_path=diff_model_path, dev=device, class_cond=args.class_cond)
        diff_model_proto.eval()
        if args.class_cond:
            print("Using class conditional diffusion model")
            diff_model = partial(diff_model_proto.forward, y=classes)
        classifier = load_guided_classifier(model_path=class_model_path, dev=device, image_size=image_size)
        classifier.eval()

    betas, time_steps = respaced_beta_schedule(
        original_betas=beta_schedule(num_timesteps=args.num_diff_steps),
        T=args.num_diff_steps,
        respaced_T=args.respaced_num_diff_steps,
    )
    diff_sampler = DiffusionSampler(betas=betas, time_steps=time_steps, posterior_variance=var_mode)
    diff_sampler.to(device)
    mcmc_steps = 4
    # step_sizes = diff_sampler.betas * 0.005
    # step_sizes[70:400] = diff_sampler.betas[70:400] * 0.001
    # step_sizes[:70] = diff_sampler.betas[:70] * 0.0001
    # a = 0.1
    # b = 1.8
    a = 0.05
    b = 1.6
    step_sizes = a * diff_sampler.betas**b
    mcmc_sampler = AnnealedHMCScoreSampler(mcmc_steps, step_sizes, 0.9, diff_sampler.betas, 3, None)
    guidance = ClassifierFullGuidance(classifier, lambda_=args.guid_scale)
    guided_sampler = MCMCGuidanceSampler(
        diff_model=diff_model,
        diff_proc=diff_sampler,
        guidance=guidance,
        mcmc_sampler=mcmc_sampler,
        reverse=True,
    )
    num_samples = args.num_samples
    th.manual_seed(0)
    classes = th.randint(10, (num_samples,), dtype=th.int64)
    print("Sampling...")
    samples, _ = guided_sampler.sample(
        num_samples, classes, device, th.Size((channels, image_size, image_size)), verbose=True
    )

    data = dict()
    data["samples"] = samples.detach().cpu()
    data["accepts"] = guided_sampler.mcmc_sampler.accepts
    data["parameters"] = {"stepsizes": step_sizes.detach().cpu(), "a": a, "b": b}
    data["classes"] = classes
    save_file = Path.cwd() / "outputs" / f"cfg_{args.diff_model}.p"
    print(f"Saving samples to '{save_file}'")
    pickle.dump(data, open(save_file, "wb"))


def _load_class(class_path: Path, device):
    classifier = load_classifier(class_path, True)
    classifier.to(device)
    classifier.eval()
    return classifier


def parse_args():
    parser = ArgumentParser(prog="Sample with MCMC classifier-full guidance")
    parser.add_argument("--guid_scale", default=1.0, type=float, help="Guidance scale")
    parser.add_argument("--num_samples", default=100, type=int, help="Num samples (batch size to run in parallell)")
    parser.add_argument("--num_diff_steps", default=1000, type=int, help="Num diffusion steps")
    parser.add_argument(
        "--respaced_num_diff_steps",
        default=250,
        type=int,
        help="Number of respaced diffusion steps (fewer than or equal to num_diff_steps)",
    )
    parser.add_argument("--diff_model", type=str, help="Diffusion model file (withouth '.pt' extension)")
    parser.add_argument("--class_model", type=str, help="Classifier model file (withouth '.pt' extension)")
    parser.add_argument("--class_cond", action="store_true", help="Use classconditional diff. model")
    parser.add_argument("--plot", action="store_true", help="enables plots")
    return parser.parse_args()


if __name__ == "__main__":
    main()
