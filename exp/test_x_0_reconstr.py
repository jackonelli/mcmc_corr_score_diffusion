"""Script for sampling with reconstruction guidance"""
from argparse import ArgumentParser
from pathlib import Path
import torch as th
import torch.nn.functional as F
from src.guidance.reconstruction import ReconstructionGuidance, ReconstructionSampler
from src.model.resnet import load_classifier
from src.utils.net import Device, get_device
from src.diffusion.base import DiffusionSampler
from src.diffusion.beta_schedules import improved_beta_schedule
from src.model.unet import load_mnist_diff
from src.utils.vis import every_nth_el, plot_diff_seq, plot_reconstr_diff_seq


@th.no_grad()
def main():
    device = get_device(Device.GPU)
    models_dir = Path.cwd() / "models"
    uncond_diff = load_mnist_diff(models_dir / "uncond_unet_mnist.pt", device)
    classifier = _load_class(models_dir / "resnet_reconstruction_classifier_mnist.pt", device)
    T = 1000
    diff_sampler = DiffusionSampler(improved_beta_schedule, num_diff_steps=T)
    diff_sampler.to(device)
    num_samples = 1
    _, samples = diff_sampler.sample(uncond_diff, num_samples, device, th.Size((1, 28, 28)))

    guidance = ReconstructionGuidance(uncond_diff, classifier, diff_sampler.alphas_bar.clone(), F.cross_entropy)
    print("Reconstructing x_0")

    # samples = 10 * [(12, th.randn((1, 1, 28, 28)).to(device))]
    samples = every_nth_el(samples, every_nth=100)
    pred_samples = predict_x_0(samples, guidance, device)
    print(len(samples))
    print(len(pred_samples))
    plot_reconstr_diff_seq(samples, pred_samples)


def predict_x_0(samples, guidance, device):
    predicted_samples = []
    for t, x_t in samples:
        x_t = x_t.to(device)
        x_0_hat = guidance.predict_x_0(x_t, t)
        predicted_samples.append((t, x_0_hat))
    return predicted_samples


def _load_class(class_path: Path, device):
    classifier = load_classifier(class_path)
    classifier.to(device)
    classifier.eval()
    return classifier


# def parse_args():
#     parser = ArgumentParser(prog="Sample with reconstruction guidance")
#     parser.add_argument("--guid_scale", default=1.0, type=float, help="Guidance scale")
#     return parser.parse_args()


if __name__ == "__main__":
    main()
