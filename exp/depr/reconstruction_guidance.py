"""Script for sampling with reconstruction guidance"""


from argparse import ArgumentParser
from pathlib import Path
import torch as th
from src.guidance.reconstruction import ReconstructionGuidance
from src.guidance.base import GuidanceSampler
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
    classifier = _load_class(models_dir / "resnet_reconstruction_classifier_mnist.pt", device)
    T = 1000
    betas = improved_beta_schedule(num_timesteps=T)
    time_steps = th.tensor([i for i in range(T)])
    diff_sampler = DiffusionSampler(betas, time_steps)
    diff_sampler.to(device)

    guidance = ReconstructionGuidance(uncond_diff, classifier, diff_sampler.alphas_bar.clone(), lambda_=args.guid_scale)
    reconstr_guided_sampler = GuidanceSampler(uncond_diff, diff_sampler, guidance)

    num_samples = 100
    classes = th.ones((num_samples,), dtype=th.int64)
    samples, _ = reconstr_guided_sampler.sample(num_samples, classes, device, th.Size((1, 28, 28)))
    plot_samples_grid(samples.detach().cpu())


def _load_class(class_path: Path, device):
    classifier = load_classifier(class_path)
    classifier.to(device)
    classifier.eval()
    return classifier


def parse_args():
    parser = ArgumentParser(prog="Sample with reconstruction guidance")
    parser.add_argument("--guid_scale", default=1.0, type=float, help="Guidance scale")
    return parser.parse_args()


if __name__ == "__main__":
    main()
