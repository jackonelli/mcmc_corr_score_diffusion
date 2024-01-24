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
from src.data.cifar import CIFAR_100_NUM_CLASSES, CIFAR_NUM_CHANNELS

DEVICE = get_device(Device.GPU)
CHANNELS, IMAGE_SIZE = 3, 256
BATCH_SIZE = 5
GUIDANCE_CLASSIFIER_PATH = Path.cwd() / f"models/{IMAGE_SIZE}x{IMAGE_SIZE}_classifier.pt"


def main():
    args = parse_args()
    # Setup and assign a directory where simulation results are saved.
    sim_dirs = collect_sim_dirs(args.res_dir)
    pattern = SimPattern(args.method, args.guid_scale, args.step_factor)
    res = compute_metrics(sim_dirs, pattern, args.classifier, args.dataset)
    format_metrics(res)
    save_metrics(res, args.store_dir, args.classifier)


def save_metrics(res: List[Tuple[float, float, SimulationConfig, int, str]], dir_: Path, classifier: str):
    compiled_res = []
    for acc, r3_acc, config, num_samples, sim_dir in res:
        metric = {"config": config, "acc": acc, "r3_acc": r3_acc, "num_samples": num_samples, "sim_dir": sim_dir}
        compiled_res.append(metric)
    dir_.mkdir(exist_ok=True, parents=True)
    with open(dir_ / f"r3_{classifier}.p", "wb") as ff:
        pickle.dump(compiled_res, ff)


def format_metrics(res: List[Tuple[float, float, SimulationConfig, int, str]]):
    for acc, r3_acc, config, num_samples, sim_dir in res:
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
        print(f"Acc: {acc}, R3 acc: {r3_acc}, with {num_samples} samples")
        print(50 * "-")


def compute_metrics(sim_dirs, pattern, classifier, dataset):
    classifier, transform = load_classifier(classifier, dataset)
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
        simple_acc, r3_acc, num_samples = compute_acc(classifier, classes_and_samples, transform, BATCH_SIZE, DEVICE)
        res.append((simple_acc, r3_acc, config, num_samples, sim_dir.name))
    return res


def load_classifier(arch: str, dataset: str):
    if dataset == "imagenet":
        classifier, transform = load_classifier_imagenet(arch, IMAGE_SIZE)
    elif dataset == "cifar100":
        classifier, transform = load_classifier_cifar()
    else:
        raise ValueError(f"Incorrect data set {dataset}")

    classifier.eval()
    ts = th.zeros((BATCH_SIZE,)).to(DEVICE)
    classifier = partial(classifier, t=ts)
    return classifier, transform


def load_classifier_imagenet(classifier: str, image_size: int):
    if classifier == "guidance":
        classifier_path = GUIDANCE_CLASSIFIER_PATH
        assert classifier_path.exists(), f"Model '{classifier_path}' does not exist."
        class_ = load_guided_classifier(model_path=classifier_path, dev=DEVICE, image_size=image_size)
        # Return dummy unit transform
        return class_, lambda x: x
    elif classifier == "independent":
        # https://pytorch.org/vision/stable/models/generated/torchvision.models.regnet_x_8gf.html
        class_ = regnet_x_8gf(weights=RegNet_X_8GF_Weights.IMAGENET1K_V2)
        class_.to(DEVICE)
        return class_, CLASSIFIER_TRANSFORM
    else:
        raise ValueError("Incorrect classifier type")


def load_classifier_cifar():
    print("Using ResNet classifier")
    classifier = load_resnet_classifier_t(
        model_path=Path.cwd() / "models/cifar100_resnet_class_t.ckpt",
        dev=DEVICE,
        emb_dim=112,
        num_classes=CIFAR_100_NUM_CLASSES,
        num_channels=CIFAR_NUM_CHANNELS,
    ).to(DEVICE)
    return classifier, lambda x: x


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
                include &= config.mcmc_stepsizes["params"]["factor"] == self.factor
            else:
                # Do not include "Reverse" if a specific factor is selected
                include = False
        return include


def parse_args():
    parser = ArgumentParser(prog="Sample from diffusion model")
    parser.add_argument("--res_dir", type=Path, required=True, help="Parent dir for all results")
    parser.add_argument("--store_dir", type=Path, default=Path.cwd() / "store/metrics", help="Dir to sort tables to")
    parser.add_argument("--metric", default="all", type=str, choices=["all", "acc", "fid"], help="Metric to compute")
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
