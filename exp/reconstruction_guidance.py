"""Script for sampling with reconstruction guidance"""


from pathlib import Path
import torch as th
import torch.nn.functional as F
from src.guidance.reconstruction import ReconstructionGuidance, ReconstructionSampler
from src.model.resnet import load_classifier
from src.utils.net import Device, get_device
from src.diffusion.base import DiffusionSampler
from src.diffusion.beta_schedules import improved_beta_schedule
from src.model.unet import load_mnist_diff
from src.utils.vis import plot_samples


def main():
    device = get_device(Device.GPU)
    models_dir = Path.cwd() / "models"
    uncond_diff = load_mnist_diff(models_dir / "uncond_unet_mnist.pt", device)
    classifier = _load_class(models_dir / "resnet_reconstruction_classifier_mnist.pt", device)
    T = 100
    diff_sampler = DiffusionSampler(improved_beta_schedule, num_diff_steps=T)
    diff_sampler.to(device)
    guidance = ReconstructionGuidance(
        uncond_diff, classifier, diff_sampler.alphas_bar.clone(), F.cross_entropy, lambda_=0.0
    )
    reconstr_guided_sampler = ReconstructionSampler(uncond_diff, diff_sampler, guidance)

    num_samples = 100
    classes = th.ones((num_samples,), dtype=th.int64)
    samples, _ = reconstr_guided_sampler.sample(num_samples, classes, device, th.Size((1, 28, 28)))
    plot_samples(samples)


def _load_class(class_path: Path, device):
    classifier = load_classifier(class_path)
    classifier.to(device)
    classifier.eval()
    return classifier


if __name__ == "__main__":
    main()
