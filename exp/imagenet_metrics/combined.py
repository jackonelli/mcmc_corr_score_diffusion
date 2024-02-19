"""Compute metrics on sampled images

From the simulation cluster we get many separate files for a simulation,
i.e., one per batch (these are name `samples_<sim_id>_<batch_id>.th).

This script iterates over a collection of such experiments and computes metrics.

Example dir. structure:
    res_dir
        |- score_rev
            |- config.json
            |- samples_0_0.th
            |- classes_0_0.th
            |- ...
        |- score_hmc
            |- config.json
            |- samples_0_0.th
            |- classes_0_0.th
            |- ...
The script will iterate over subdirs in `args.res_dir` and compute separate metrics for `score_rev` and `score_hmc` in this example.


You can also filter the subdirs on params such as method, guidance scale, step length factor.
"""
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
from src.model.cifar.standard_class import load_standard_class

DEVICE = get_device(Device.GPU)
CHANNELS, IMAGE_SIZE = 3, 256
GUIDANCE_CLASSIFIER_PATH = Path.cwd() / f"models/{IMAGE_SIZE}x{IMAGE_SIZE}_classifier.pt"


def main():
    args = parse_args()
    sim_dirs = collect_sim_dirs(args.res_dir)
    pattern = SimPattern(args.method, args.param, args.guid_scale, args.step_factor)
    res = compute_metrics(sim_dirs, pattern, args.classifier, args.dataset, args.batch_size)
    format_metrics(res)
    save_metrics(res, dataset=args.dataset, param=args.param, dir_=args.store_dir)


def compute_metrics(sim_dirs, pattern, classifier, dataset, batch_size):
    classifier, transform = load_classifier(classifier, dataset, batch_size)
    res = []
    for sim_dir in sim_dirs:
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
    filename = dir_ / f"{dataset}_{param}.p"
    with open(filename, "wb") as ff:
        pickle.dump(compiled_res, ff)
    print(f"Metrics written to {filename.relative_to(filename.parent.parent.parent)}")


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


def load_classifier(type_: str, dataset: str, batch_size: int):
    if dataset == "imagenet":
        classifier, transform = load_classifier_imagenet(type_, IMAGE_SIZE, batch_size)
    elif dataset == "cifar100":
        classifier, transform = load_classifier_cifar(type_, batch_size)
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


def load_classifier_cifar(type_: str, batch_size: int):
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


def collect_sim_dirs(res_dir: Path):
    """Create an iterator over valid sub dirs (those with config and samples)"""
    dirs = filter(lambda x: x.is_dir(), res_dir.iterdir())
    dirs = filter(lambda x: (x / "config.json").exists(), dirs)
    dirs = filter(lambda x: len(list(x.glob("classes*"))) > 0, dirs)
    dirs = filter(lambda x: len(list(x.glob("samples*"))) > 0, dirs)
    # Filter dirs with only the config file will be present (i.e, no samples)
    dirs = filter(lambda x: len(list(x.glob("*"))) > 1, dirs)
    return dirs


def collect_samples(sim_dir: Path) -> Tuple[List[Tuple[Path, Path]], int]:
    """Collect sampled images and corresponding classes

    Results are saved to timestamped dirs to prevent overwriting, we need to collect samples from all dirs with matching sim name.
    """
    classes = sorted([path.name for path in sim_dir.glob("classes_*_*.th")])
    classes = [sim_dir / cl for cl in classes]
    samples = sorted([path.name for path in sim_dir.glob("samples_*_*.th")])
    samples = [sim_dir / sm for sm in samples]
    assert len(samples) == len(classes)
    coll = list(zip(classes, samples))
    validate(coll)
    num_batches = len(samples)
    return coll, num_batches


def validate(coll):
    """Double check that the sorting shenanigans above actually pair the correct samples to the correct classes"""
    for cl, sm in coll:
        tmp = PATTERN.match(cl.name)
        assert tmp is not None, f"No match for sim or batch id in '{cl.name}'"
        cl_sim, cl_batch = tmp[1], tmp[2]
        tmp = PATTERN.match(sm.name)
        assert tmp is not None, f"No match for sim or batch id in '{sm.name}'"
        sm_sim, sm_batch = tmp[1], tmp[2]
        assert cl_sim == sm_sim, "Mismatched slurm ID"
        assert cl_batch == sm_batch, "Mismatched batch index"


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
                include = self.method == "all" or self.method == "rev"
        include &= self.param in config.diff_model if not "256" in config.diff_model else True
        return include


def parse_args():
    parser = ArgumentParser(prog="Sample from diffusion model")
    parser.add_argument("--res_dir", type=Path, required=True, help="Parent dir for all results")
    parser.add_argument("--store_dir", type=Path, default=Path.cwd() / "results/metrics", help="Dir to save tables to")
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
        "--method",
        default="all",
        type=str,
        choices=["all", "rev", "ula", "la", "uhmc", "hmc"],
        help="MCMC methods to compute metrics for",
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
