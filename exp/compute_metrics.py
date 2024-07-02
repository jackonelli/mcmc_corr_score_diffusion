"""Compute metrics on sampled images"""
import sys
sys.path.append(".")

sys.path.append("imagenet_metrics")
from dataclasses import dataclass
from pathlib import Path
from argparse import ArgumentParser
from typing import Tuple, List
from functools import partial
import pickle
import os
import torch as th
from torchvision.models import regnet_x_8gf, RegNet_X_8GF_Weights
from src.data.imagenet import CLASSIFIER_TRANSFORM
from src.utils.net import get_device, Device
from src.model.guided_diff.classifier import load_guided_classifier
from exp.utils import SimulationConfig, UnguidedSimulationConfig
from exp.imagenet_metrics.common import PATTERN, compute_acc, compute_nbr_samples

# Cifar
from src.model.resnet import load_classifier_t as load_resnet_classifier_t
from src.data.cifar import CIFAR_100_NUM_CLASSES, CIFAR_NUM_CHANNELS, CIFAR_10_NUM_CLASSES, CIFAR_IMAGE_SIZE
from src.model.cifar.standard_class import load_standard_class, vgg13_bn, vgg16_bn
from src.utils.fid_utils import get_model
from pytorch_fid.fid_score import calculate_frechet_distance
from src.model.cifar.utils import load_unet_ho_drop_classifier_t
from exp.compute_fid import get_statistics
from torchvision.transforms import (
    Compose,
    Normalize,
    Lambda,
    ToTensor
)

DEVICE = get_device(Device.GPU)
CHANNELS, IMAGE_SIZE = 3, 256
GUIDANCE_CLASSIFIER_PATH = Path.cwd() / f"models/{IMAGE_SIZE}x{IMAGE_SIZE}_classifier.pt"


def main():
    args = parse_args()
    # Setup and assign a directory where simulation results are saved.
    sim_dirs = collect_sim_dirs(args.res_dir)
    p_file = collect_computed_res(args.res_dir, args.param)
    pattern = SimPattern(args.method, args.param, args.guid_scale, args.step_factor)
    res_acc, res_fid = compute_metrics(p_file, sim_dirs, pattern, args.classifier, args.dataset, args.batch_size, args.path_fid,
                                       args.num_samples)
    format_metrics(res_acc, res_fid)
    if not args.no_save:
        save_metrics(res_acc, res_fid, dataset=args.dataset, param=args.param, dir_=args.store_dir)


def save_metrics(
    res_acc: List[Tuple[float, float, float, SimulationConfig, int, str]],
    res_fid: List[Tuple[float]],
    dataset: str,
    param: str,
    dir_: Path,
):
    compiled_res = []
    for (acc, r3_acc, top_5_acc, config, num_samples, sim_dir), (fid,) in zip(res_acc, res_fid):
        metric = {
            "config": config,
            "acc": acc,
            "r3_acc": r3_acc,
            "top_5_acc": top_5_acc,
            "num_samples": num_samples,
            "sim_dir": sim_dir,
            "fid": fid,
        }
        compiled_res.append(metric)
    dir_.mkdir(exist_ok=True, parents=True)
    pickle.dump(compiled_res, open( dir_ / f"{dataset}_{param}.p", "wb" ))
    # with open(dir_ / f"{dataset}_{param}.p", "wb") as ff:
    #    pickle.dump(compiled_res, ff)


def format_metrics(res: List[Tuple[float, float, float, SimulationConfig, int, str]], res_fid: List[Tuple[float]]):
    for (acc, r3_acc, top_5_acc, config, num_samples, sim_dir), (fid,) in zip(res, res_fid):
        if isinstance(config, SimulationConfig):
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
        else:
            str_ = f"Rev"
        print(f"\n{sim_dir}")
        print(str_)
        print(f"Acc: {acc}, R3 acc: {r3_acc}, Top-5 acc: {top_5_acc}, FID: {fid} with {num_samples} samples")
        print(50 * "-")


def compute_metrics(p_file, sim_dirs, pattern, classifier, dataset, batch_size, path_fid_compare=None, n_max=None):
    classifier, transform = load_classifier(classifier, dataset, batch_size)
    dims = 2048
    m1, s1, fid_model = None, None, None
    if path_fid_compare is not None:
        fid_model = get_model(DEVICE, dims)
        m1, s1 = get_statistics(model=fid_model,
                                device=DEVICE,
                                batch_size=batch_size,
                                dims=dims,
                                path_dataset=path_fid_compare,
                                type_dataset='stats',
                                num_workers=8,
                                path_save_stats=False)
    res_acc = []
    res_fid = []

    if p_file is not None:
        computed_dirs = [f['sim_dir'] for f in p_file]
    else:
        computed_dirs = []

    for sim_dir in sim_dirs:
        num_files = len(list(sim_dir.iterdir()))
        if num_files == 1:
            print(f"Skipping dir '{sim_dir.name}', with no samples")
            continue

        config = SimulationConfig.load(sim_dir / "config.json")
        if 'unguided' in config['name']:
            config = UnguidedSimulationConfig.from_json_no_load(config)
        else:
            config = SimulationConfig.from_json_no_load(config)
        if not pattern.include_sim(config):
            continue

        if sim_dir.name in computed_dirs:
            print(f"The directory '{sim_dir.name}' has been processed earlier - load results")
            idx = computed_dirs.index(sim_dir.name)
            comp_res = p_file[idx]
            simple_acc, r3_acc, top_5_acc, num_samples, fid_value = (
                comp_res['acc'], comp_res['r3_acc'], comp_res['top_5_acc'], comp_res['num_samples'], comp_res['fid']
            )
            res_acc.append((simple_acc, r3_acc, top_5_acc, config, num_samples, sim_dir.name))
            res_fid.append((fid_value,))
        else:
            classes_and_samples, num_batches = collect_samples(sim_dir)
            if isinstance(config, SimulationConfig):
                print(f"Processing '{sim_dir.name}' with {num_batches} batches")
                simple_acc, r3_acc, top_5_acc, num_samples = compute_acc(
                    classifier, classes_and_samples, transform, batch_size, DEVICE, n_max
                )
            else:
                num_samples = compute_nbr_samples(classes_and_samples)
                simple_acc, r3_acc, top_5_acc = "-", "-", "-"

            res_acc.append((simple_acc, r3_acc, top_5_acc, config, num_samples, sim_dir.name))
            if path_fid_compare is not None:
                m2, s2 = get_statistics(model=fid_model,
                                        device=DEVICE,
                                        batch_size=batch_size,
                                        dims=dims,
                                        path_dataset=sim_dir,
                                        type_dataset='th',
                                        num_workers=8,
                                        path_save_stats=None,
                                        num_samples=n_max)
                if m2 is None or s2 is None:
                    fid_value = th.inf
                else:
                    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
                res_fid.append((fid_value,))
            else:
                res_fid.append(('-',))
    return res_acc, res_fid


def load_classifier(type_: str, dataset: str, batch_size: int):
    if dataset == "imagenet":
        classifier, transform = load_classifier_imagenet(type_, IMAGE_SIZE, batch_size)
    elif dataset == "cifar100":
        classifier, transform = load_classifier_cifar100(type_, batch_size)
    elif dataset == "cifar10":
        classifier, transform = load_classifier_cifar10(type_, batch_size)
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
        classifier = vgg16_bn(model_path=Path.cwd() / "models/vgg16_bn_ema_cifar100.ckpt",
                              dataset='cifar100').to(DEVICE)
        classifier.eval()
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
        classifier = vgg13_bn(model_path=Path.cwd() / "models/vgg13_bn_cifar10.pt").to(DEVICE)
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


def collect_computed_res(res_dir: Path, param: str):
    pickle_file = [f for f in os.listdir(res_dir) if ".p" in f and param in f]
    if len(pickle_file) > 0:
        file = pickle.load(open(res_dir / pickle_file[0], "rb"))
    else:
        file = None
    return file


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
        # print(cl.name)
        cl_1, cl_2 = tmp[1], tmp[2]
        tmp = PATTERN.match(sm.name)
        # print(sm.name)
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
        include &= self.param in config.diff_model or self.param == 'both'
        return include


def parse_args():
    parser = ArgumentParser(prog="Sample from diffusion model")
    parser.add_argument("--res_dir", type=Path, required=True, help="Parent dir for all results")
    parser.add_argument("--store_dir", type=Path, default=Path.cwd() / "results/cifar100", help="Dir to sort tables to")
    parser.add_argument("--metric", default="all", type=str, choices=["all", "acc", "fid"], help="Metric to compute")
    parser.add_argument(
        "--param", default="both", type=str, choices=["energy", "score", "both"], help="Choose diff model parameterisation"
    )
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size")
    parser.add_argument(
        "--dataset", default="cifar100", type=str, choices=["imagenet", "cifar100", "cifar10"], help="Choose dataset"
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
    parser.add_argument("--path_fid", default=None, type=str, help="Path to fid-stats of data sets")
    parser.add_argument('--num_samples', default=None, type=int, help='Number of samples to evaluate. '
                                                                      'If None, then all samples are evaluated.')
    parser.add_argument('--no_save', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    main()
