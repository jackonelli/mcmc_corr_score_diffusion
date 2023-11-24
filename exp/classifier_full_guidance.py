"""Script for sampling with reconstruction guidance"""


import sys


sys.path.append(".")
from argparse import ArgumentParser
from pathlib import Path
from functools import partial
import torch as th
from src.guidance.base import GuidanceSampler
from src.guidance.classifier_full import ClassifierFullGuidance
from src.model.resnet import load_classifier
from src.utils.net import Device, get_device
from src.diffusion.base import DiffusionSampler
from src.diffusion.beta_schedules import improved_beta_schedule, linear_beta_schedule
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
    print("Loading models")
    if "mnist" in args.diff_model:
        channels, image_size = 1, 28
        diff_model = load_mnist_diff(diff_model_path, device)
        classifier = _load_class(models_dir / class_model_path, device)
    elif "256x256_diffusion" in args.diff_model:
        channels, image_size = 3, 256
        diff_model_proto = load_guided_diff_unet(model_path=diff_model_path, dev=device, class_cond=args.class_cond)
        diff_model_proto.eval()
        if args.class_cond:
            print("Using class conditional diffusion model")
            diff_model = partial(diff_model_proto.forward, y=classes)
        classifier = load_guided_classifier(model_path=class_model_path, dev=device, image_size=image_size)
        classifier.eval()

    T = args.num_diff_steps
    diff_sampler = DiffusionSampler(linear_beta_schedule, num_diff_steps=T, posterior_variance="learned")
    diff_sampler.to(device)

    guidance = ClassifierFullGuidance(classifier, lambda_=args.guid_scale)
    guid_sampler = GuidanceSampler(diff_model, diff_sampler, guidance, verbose=True)

    print("Sampling...")
    samples, _ = guid_sampler.sample(num_samples, classes, device, th.Size((channels, image_size, image_size)))
    save_file = Path.cwd() / "outputs" / f"cfg_{args.diff_model}.th"
    print(f"Saving samples to '{save_file}'")
    th.save(samples, save_file)


def _load_class(class_path: Path, device):
    classifier = load_classifier(class_path, True)
    classifier.to(device)
    classifier.eval()
    return classifier


def parse_args():
    parser = ArgumentParser(prog="Sample with classifier-full guidance")
    parser.add_argument("--guid_scale", default=1.0, type=float, help="Guidance scale")
    parser.add_argument("--num_samples", default=100, type=int, help="Num samples (batch size to run in parallell)")
    parser.add_argument("--num_diff_steps", default=1000, type=int, help="Num diffusion steps")
    parser.add_argument("--diff_model", type=str, help="Diffusion model file (withouth '.pt' extension)")
    parser.add_argument("--class_model", type=str, help="Classifier model file (withouth '.pt' extension)")
    parser.add_argument("--class_cond", action="store_true", help="Use classconditional diff. model")
    parser.add_argument("--plot", action="store_true", help="enables plots")
    return parser.parse_args()


if __name__ == "__main__":
    main()
