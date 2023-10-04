"""Script for sampling with classifier-full guidance with MCMC"""

from argparse import ArgumentParser
from pathlib import Path
import torch as th
from src.guidance.base import MCMCGuidanceSampler
from src.guidance.classifier_full import ClassifierFullGuidance
from src.samplers.mcmc import AnnealedHMCScoreSampler
from src.model.resnet import load_classifier
from src.utils.net import Device, get_device
from src.diffusion.base import DiffusionSampler
from src.diffusion.beta_schedules import improved_beta_schedule
from src.model.unet import load_mnist_diff
from src.utils.vis import plot_samples_grid


def main():
    args = parse_args()
    device = get_device(Device.GPU)
    models_dir = Path.cwd() / "models"
    uncond_diff = load_mnist_diff(models_dir / "uncond_unet_mnist.pt", device)
    classifier = _load_class(models_dir / "resnet_classifier_t_mnist.pt", device)
    T = 1000
    diff_sampler = DiffusionSampler(improved_beta_schedule, num_diff_steps=T)
    diff_sampler.to(device)

    mcmc_steps = 3
    mcmc_sampler = AnnealedHMCScoreSampler(mcmc_steps, diff_sampler.betas*0.01, 0.9, diff_sampler.betas, 3, None)
    guidance = ClassifierFullGuidance(classifier, lambda_=args.guid_scale)
    reconstr_guided_sampler = MCMCGuidanceSampler(diff_model=uncond_diff, diff_proc=diff_sampler, guidance=guidance,
                                                  mcmc_sampler=mcmc_sampler, reverse=True, verbose=True)

    num_samples = 100
    classes = th.ones((num_samples,), dtype=th.int64)
    samples, _ = reconstr_guided_sampler.sample(num_samples, classes, device, th.Size((1, 28, 28)))
    import pickle
    pickle.dump(samples.detach().cpu(), open("samples_mnist_mcmc_reverse_lambda1.p", "wb"))
    pickle.dump(reconstr_guided_sampler.mcmc_sampler.accepts, open("accepts.p", "wb"))
    # plot_samples_grid(samples.detach().cpu())


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
    # import pickle
    # samples = pickle.load(open("samples_mnist_mcmc.p", "rb"))
    # plot_samples_grid(samples)
    main()
