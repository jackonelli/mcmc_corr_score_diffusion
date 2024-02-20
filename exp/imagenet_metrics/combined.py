"""Compute metrics on sampled images"""
import sys


sys.path.append(".")
from dataclasses import dataclass
from pathlib import Path
from argparse import ArgumentParser
from typing import Tuple, List
from functools import partial
import pickle
import torch as th
from torchvision.models import regnet_x_8gf, RegNet_X_8GF_Weights
from src.data.imagenet import CLASSIFIER_TRANSFORM
from src.utils.net import get_device, Device
from src.model.guided_diff.classifier import load_guided_classifier
from exp.utils import SimulationConfig
from exp.imagenet_metrics.common import PATTERN, compute_acc

# Cifar
from src.model.resnet import load_classifier_t as load_resnet_classifier_t
from src.data.cifar import CIFAR_100_NUM_CLASSES, CIFAR_NUM_CHANNELS, CIFAR_10_NUM_CLASSES, CIFAR_IMAGE_SIZE
from src.model.cifar.standard_class import load_standard_class, vgg13_bn
from src.model.cifar.utils import load_unet_ho_drop_classifier_t
from torchvision.transforms import (
    Compose,
    Normalize,
    Lambda,
)

DEVICE = get_device(Device.GPU)
CHANNELS, IMAGE_SIZE = 3, 256
GUIDANCE_CLASSIFIER_PATH = Path.cwd() / f"models/{IMAGE_SIZE}x{IMAGE_SIZE}_classifier.pt"


def main():
    args = parse_args()
    # Setup and assign a directory where simulation results are saved.
    sim_dirs = collect_sim_dirs(args.res_dir)
    pattern = SimPattern(args.method, args.param, args.guid_scale, args.step_factor)
    res = compute_metrics(sim_dirs, pattern, args.classifier, args.dataset, args.batch_size)
    format_metrics(res)
    save_metrics(res, dataset=args.dataset, param=args.param, dir_=args.store_dir)


def save_metrics(
    res: List[Tuple[float, float, float, SimulationConfig, int, str]],
    dataset: str,
    param: str,
    dir_: Path,
):
    compiled_res = []
    for acc, r3_acc, top_5_acc, config, num_samples, sim_dir in res:
        metric = {
            "config": config,
            "acc": acc,
            "r3_acc": r3_acc,
            "top_5_acc": top_5_acc,
            "num_samples": num_samples,
            "sim_dir": sim_dir,
        }
        compiled_res.append(metric)
    dir_.mkdir(exist_ok=True, parents=True)
    with open(dir_ / f"{dataset}_{param}.p", "wb") as ff:
        pickle.dump(compiled_res, ff)


def format_metrics(res: List[Tuple[float, float, float, SimulationConfig, int, str]]):
    for acc, r3_acc, top_5_acc, config, num_samples, sim_dir in res:
        lambda_ = config.guid_scale
        if config.mcmc_method is None:
            str_ = f"Rev, lambda={lambda_}"
        else:
            mcmc_method = config.mcmc_method if config.mcmc_method is not None else "Rev"
            mcmc_steps = config.mcmc_steps
            mcmc_lower_t = config.mcmc_lower_t if config.mcmc_lower_t is not None else 0
            n_trapets = config.n_trapets
            step_params = config.mcmc_stepsizes["params"]
            step_sch = config.mcmc_stepsizes["beta_schedule"]
            factor, exp = step_params["factor"], step_params["exponent"]
            str_ = f"{mcmc_method.upper()}-{mcmc_steps}({mcmc_lower_t}), lambda={lambda_}, n_trapets={n_trapets}, {factor} * beta_{step_sch}^{exp}"
        print(f"\n{sim_dir}")
        print(str_)
        print(f"Acc: {acc}, R3 acc: {r3_acc}, Top-5 acc: {top_5_acc}, with {num_samples} samples")
        print(50 * "-")


def compute_metrics(sim_dirs, pattern, classifier, dataset, batch_size):
    classifier, transform = load_classifier(classifier, dataset, batch_size)
    res = []
    for sim_dir in sim_dirs:
        num_files = len(list(sim_dir.iterdir()))
        if num_files == 1:
            print(f"Skipping dir '{sim_dir.name}', with no samples")
            continue
        config = SimulationConfig.from_json(sim_dir / "config.json")
        if not pattern.include_sim(config):
            continue
        classes_and_samples, num_batches = collect_samples(sim_dir)
        print(f"Processing '{sim_dir.name}' with {num_batches} batches")
        simple_acc, r3_acc, top_5_acc, num_samples = compute_acc(
            classifier, classes_and_samples, transform, batch_size, DEVICE
        )
        res.append((simple_acc, r3_acc, top_5_acc, config, num_samples, sim_dir.name))
    return res


def load_classifier(type_: str, dataset: str, batch_size: int):
    if dataset == "imagenet":
        classifier, transform = load_classifier_imagenet(type_, IMAGE_SIZE, batch_size)
    elif dataset == "cifar100":
        classifier, transform = load_classifier_cifar100(type_, batch_size)
    else:
        raise ValueError(f"Incorrect data set {dataset}")

    return classifier, transform


def load_classifier_imagenet(type_: str, image_size: int, batch_size: int):
    if type_ == "guidance":
        classifier_path = GUIDANCE_CLASSIFIER_PATH
        assert classifier_path.exists(), f"Model '{classifier_path}' does not exist."
        classifier = load_guided_classifier(model_path=classifier_path, dev=DEVICE, image_size=image_size)
        classifier.eval()
        # Remove time-dependency
        ts = th.zeros((batch_size,)).to(DEVICE)
        classifier = partial(classifier, t=ts)
        # Return dummy unit transform
        transform = lambda x: x
    elif type_ == "independent":
        # https://pytorch.org/vision/stable/models/generated/torchvision.models.regnet_x_8gf.html
        classifier = regnet_x_8gf(weights=RegNet_X_8GF_Weights.IMAGENET1K_V2)
        classifier.eval()
        classifier.to(DEVICE)
        transform = CLASSIFIER_TRANSFORM
    else:
        raise ValueError("Incorrect classifier type")
    return classifier, transform


def load_classifier_cifar100(type_: str, batch_size: int):
    if type_ == "guidance":
        classifier = load_resnet_classifier_t(
            model_path=Path.cwd() / "models/cifar100_resnet_class_t.ckpt",
            dev=DEVICE,
            emb_dim=112,
            num_classes=CIFAR_100_NUM_CLASSES,
            num_channels=CIFAR_NUM_CHANNELS,
        ).to(DEVICE)
        classifier.eval()
        # Remove time-dependency
        ts = th.zeros((batch_size,)).to(DEVICE)
        classifier = partial(classifier, t=ts)
    elif type_ == "independent":
        classifier = load_standard_class(
            model_path=Path.cwd() / "models/cifar100_simple_class.pt",
            device=DEVICE,
            num_channels=CIFAR_NUM_CHANNELS,
            num_classes=CIFAR_100_NUM_CLASSES,
        )
        classifier.eval()
        ts = th.zeros((batch_size,)).to(DEVICE)
        classifier = partial(classifier, t=ts)
    else:
        raise ValueError("Incorrect classifier type")
    return classifier, lambda x: x


def load_classifier_cifar10(type_: str, batch_size: int):
    if type_ == "guidance":
        x_size = (CIFAR_NUM_CHANNELS, CIFAR_IMAGE_SIZE, CIFAR_IMAGE_SIZE)
        classifier = load_unet_ho_drop_classifier_t(
            model_path=Path.cwd() / "models/cifar10_class_t.ckpt",
            dev=DEVICE,
            dropout=0.,
            num_diff_steps=1000,
            num_classes=CIFAR_10_NUM_CLASSES,
            x_size=x_size
        ).to(DEVICE)
        classifier.eval()
        # Remove time-dependency
        ts = th.zeros((batch_size,)).to(DEVICE)
        classifier = partial(classifier, t=ts)
        transform = lambda x: x
    elif type_ == "independent":
        classifier = vgg13_bn(pretrained=True).to(DEVICE)
        classifier.eval()
        transform = Compose(
            [
                Lambda(lambda x: (x + 1) / 2),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        raise ValueError("Incorrect classifier type")
    return classifier, transform


def collect_sim_dirs(res_dir: Path):
    dirs = filter(lambda x: x.is_dir(), res_dir.iterdir())
    dirs = filter(lambda x: (x / "config.json").exists(), dirs)
    return dirs


def collect_samples(sim_dir: Path) -> Tuple[List[Tuple[Path, Path]], int]:
    """Collect sampled images and corresponding classes

    Results are saved to timestamped dirs to prevent overwriting, we need to collect samples from all dirs with matching sim name.
    """
    classes = []
    samples = []
    # assert config.name in sim_name
    classes.extend(sorted([path.name for path in sim_dir.glob("classes_*_*.th")]))
    classes = [sim_dir / cl for cl in classes]
    samples.extend(sorted([path.name for path in sim_dir.glob("samples_*_*.th")]))
    samples = [sim_dir / sm for sm in samples]
    assert len(samples) == len(classes)
    num_batches = len(samples)
    coll = list(zip(classes, samples))
    validate(coll)
    return coll, num_batches


def validate(coll):
    """Double check that the sorting shenanigans above actually pair the correct samples to the correct classes"""
    for cl, sm in coll:
        tmp = PATTERN.match(cl.name)
        cl_1, cl_2 = tmp[1], tmp[2]
        tmp = PATTERN.match(sm.name)
        sm_1, sm_2 = tmp[1], tmp[2]
        assert cl_1 == sm_1, "Mismatched slurm ID"
        assert cl_2 == sm_2, "Mismatched batch index"


@dataclass
class SimPattern:
    method: str
    param: str
    lambda_: float
    factor: float

    def include_sim(self, config) -> bool:
        # MCMC method
        if self.method == "all":
            include = True
        else:
            if config.mcmc_method is None:
                include = "rev" == self.method
            else:
                include = config.mcmc_method == self.method

        if self.lambda_ is not None:
            include &= config.guid_scale == self.lambda_

        if self.factor is not None:
            if config.mcmc_method is not None:
                cfg_factor = config.mcmc_stepsizes["params"]["factor"]
                include &= cfg_factor == self.factor
            else:
                # Include "Reverse" if a specific factor is selected
                include = True
        include &= self.param in config.diff_model
        return include


def parse_args():
    parser = ArgumentParser(prog="Sample from diffusion model")
    parser.add_argument("--res_dir", type=Path, required=True, help="Parent dir for all results")
    parser.add_argument("--store_dir", type=Path, default=Path.cwd() / "results/cifar100", help="Dir to sort tables to")
    parser.add_argument("--metric", default="all", type=str, choices=["all", "acc", "fid"], help="Metric to compute")
    parser.add_argument(
        "--param", default="score", type=str, choices=["energy", "score"], help="Choose diff model parameterisation"
    )
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size")
    parser.add_argument(
        "--dataset", default="imagenet", type=str, choices=["imagenet", "cifar100"], help="Choose dataset"
    )
    parser.add_argument(
        "--classifier",
        default="guidance",
        type=str,
        choices=["guidance", "independent"],
        help="Which model to compute acc. with.",
    )
    parser.add_argument(
        "--method", default="all", type=str, choices=["all", "ula", "la"], help="MCMC methods to compute metrics for"
    )
    parser.add_argument(
        "--guid_scale",
        default=None,
        type=float,
        help="Guid. scale (lambda) if 'None', then all guid scales are included.",
    )
    parser.add_argument(
        "--step_factor",
        default=None,
        type=float,
        help="Step factor (a) if 'None', then all a-values are included.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
