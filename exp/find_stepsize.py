"""Script for sampling with classifier-full guidance with MCMC and adaptive step size"""

from argparse import ArgumentParser
from pathlib import Path
import torch as th
from src.guidance.base import MCMCGuidanceSampler
from src.guidance.classifier_full import ClassifierFullGuidance
from src.samplers.mcmc import AnnealedHMCScoreSampler, AdaptiveStepSizeMCMCSamplerWrapper
from src.model.resnet import load_classifier
from src.utils.net import Device, get_device
from src.diffusion.base import DiffusionSampler
from src.diffusion.beta_schedules import improved_beta_schedule
from src.model.unet import load_mnist_diff
import pickle


def main():
    args = parse_args()
    device = get_device(Device.GPU)
    models_dir = Path.cwd() / "models"
    uncond_diff = load_mnist_diff(models_dir / "uncond_unet_mnist.pt", device)
    classifier = _load_class(models_dir / "resnet_classifier_t_mnist.pt", device)
    T = 1000
    betas = improved_beta_schedule(num_timesteps=T)
    time_steps = th.tensor([i for i in range(T)])
    diff_sampler = DiffusionSampler(betas, time_steps)
    diff_sampler.to(device)

    a = 1  # 0.05
    b = 1  # 1.6
    step_sizes = a * diff_sampler.betas ** b

    mcmc_steps = 1
    mcmc_sampler = AnnealedHMCScoreSampler(mcmc_steps, step_sizes, 0.9, diff_sampler.betas, 3, None)
    sampler = AdaptiveStepSizeMCMCSamplerWrapper(sampler=mcmc_sampler, accept_rate_bound=[0.6, 0.8], max_iter=10)
    guidance = ClassifierFullGuidance(classifier, lambda_=args.guid_scale)
    guided_sampler = MCMCGuidanceSampler(diff_model=uncond_diff, diff_proc=diff_sampler, guidance=guidance,
                                         mcmc_sampler=sampler, reverse=True, verbose=True)

    num_samples = 200
    th.manual_seed(0)
    classes = th.randint(10, (num_samples,), dtype=th.int64)
    samples, _ = guided_sampler.sample(num_samples, classes, device, th.Size((1, 28, 28)))
    adaptive_step_sizes = sampler.res
    pickle.dump(adaptive_step_sizes, open("adaptive_step_sizes.p", "wb"))


def _load_class(class_path: Path, device):
    classifier = load_classifier(class_path, True)
    classifier.to(device)
    classifier.eval()
    return classifier


def parse_args():
    parser = ArgumentParser(prog="Sample with classifier-full guidance")
    parser.add_argument("--guid_scale", default=1.0, type=float, help="Guidance scale")
    return parser.parse_args()


if __name__ == "__main__":
    main()
