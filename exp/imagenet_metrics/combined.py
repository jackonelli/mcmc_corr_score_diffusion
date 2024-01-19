"""Compute metrics on sampled images"""
import sys

sys.path.append(".")
from pathlib import Path
from argparse import ArgumentParser
from typing import Tuple, List, Optional
from src.utils.net import get_device, Device
from src.model.guided_diff.classifier import load_guided_classifier
from exp.utils import SimulationConfig
from exp.imagenet_metrics.common import PATTERN, compute_acc

DEVICE = get_device(Device.GPU)
CHANNELS, IMAGE_SIZE = 3, 256
BATCH_SIZE = 5
CLASSIFIER_PATH = Path.cwd() / f"models/{IMAGE_SIZE}x{IMAGE_SIZE}_classifier.pt"


def main():
    args = parse_args()
    # Setup and assign a directory where simulation results are saved.
    sim_dirs = collect_sim_dirs(args.res_dir, args.filter)
    res = compute_metrics(sim_dirs)
    format_metrics(res)


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


def compute_metrics(sim_dirs):
    classifier, classifier_name = load_classifier(CLASSIFIER_PATH, IMAGE_SIZE)
    res = []
    for sim_dir in sim_dirs:
        num_files = len(list(sim_dir.iterdir()))
        if num_files == 1:
            print(f"Skipping dir '{sim_dir.name}', with no samples")
            continue
        classes_and_samples, config, num_batches = collect_samples(sim_dir)
        print(f"Processing '{sim_dir.name}' with {num_batches} batches")
        assert config.classifier in classifier_name, "Classifier mismatch"
        simple_acc, r3_acc, num_samples = compute_acc(classifier, classes_and_samples, BATCH_SIZE, DEVICE)
        res.append((simple_acc, r3_acc, config, num_samples, sim_dir.name))
    return res


def load_classifier(classifier_path: Path, image_size: int):
    assert classifier_path.exists(), f"Model '{classifier_path}' does not exist."
    classifier = load_guided_classifier(model_path=classifier_path, dev=DEVICE, image_size=image_size)
    classifier.eval()
    return classifier, classifier_path.name


def collect_sim_dirs(res_dir: Path, pattern: Optional[str]):
    dirs = filter(lambda x: x.is_dir(), res_dir.iterdir())
    dirs = filter(lambda x: (x / "config.json").exists(), dirs)
    if pattern is not None:
        dirs = filter(lambda x: pattern in x.name, dirs)
    return dirs


def collect_samples(sim_dir: Path) -> Tuple[List[Tuple[Path, Path]], SimulationConfig, int]:
    """Collect sampled images and corresponding classes

    Results are saved to timestamped dirs to prevent overwriting, we need to collect samples from all dirs with matching sim name.
    """
    classes = []
    samples = []
    config = SimulationConfig.from_json(sim_dir / "config.json")
    # assert config.name in sim_name
    classes.extend(sorted([path.name for path in sim_dir.glob("classes_*_*.th")]))
    classes = [sim_dir / cl for cl in classes]
    samples.extend(sorted([path.name for path in sim_dir.glob("samples_*_*.th")]))
    samples = [sim_dir / sm for sm in samples]
    assert len(samples) == len(classes)
    num_batches = len(samples)
    coll = list(zip(classes, samples))
    validate(coll)
    return coll, config, num_batches


def validate(coll):
    """Double check that the sorting shenanigans above actually pair the correct samples to the correct classes"""
    for cl, sm in coll:
        tmp = PATTERN.match(cl.name)
        cl_1, cl_2 = tmp[1], tmp[2]
        tmp = PATTERN.match(sm.name)
        sm_1, sm_2 = tmp[1], tmp[2]
        assert cl_1 == sm_1, "Mismatched slurm ID"
        assert cl_2 == sm_2, "Mismatched batch index"


def parse_args():
    parser = ArgumentParser(prog="Sample from diffusion model")
    parser.add_argument("--res_dir", type=Path, required=True, help="Parent dir for all results")
    parser.add_argument("--metric", default="all", type=str, choices=["all", "acc", "fid"], help="Metric to compute")
    parser.add_argument("--filter", default=None, type=str, help="Optional filter of sub. dir names")
    return parser.parse_args()


if __name__ == "__main__":
    main()
