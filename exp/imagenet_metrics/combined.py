"""Compute metrics on sampled images"""
import sys

sys.path.append(".")
from pathlib import Path
from argparse import ArgumentParser
from typing import Tuple, List
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
    sim_dirs = collect_sim_dirs(args.res_dir)
    compute_metrics(sim_dirs)


def compute_metrics(sim_dirs):
    classifier, classifier_name = load_classifier(CLASSIFIER_PATH, IMAGE_SIZE)
    for sim_dir in sim_dirs:
        print(sim_dir.name)
        classes_and_samples, config = collect_samples(sim_dir)
        assert config.classifier in classifier_name, "Classifier mismatch"
        simple_acc, r3_acc = compute_acc(classifier, classes_and_samples, BATCH_SIZE, DEVICE)
        print(simple_acc, r3_acc)


def load_classifier(classifier_path: Path, image_size: int):
    assert classifier_path.exists(), f"Model '{classifier_path}' does not exist."
    classifier = load_guided_classifier(model_path=classifier_path, dev=DEVICE, image_size=image_size)
    classifier.eval()
    return classifier, classifier_path.name


def collect_sim_dirs(res_dir: Path):
    dirs = filter(lambda x: x.is_dir(), res_dir.iterdir())
    dirs = filter(lambda x: (x / "config.json").exists(), dirs)
    return dirs


def collect_samples(sim_dir: Path) -> Tuple[List[Tuple[Path, Path]], SimulationConfig]:
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
    print(f"Found {len(samples)} samples")
    coll = list(zip(classes, samples))
    validate(coll)
    return coll, config


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
    return parser.parse_args()


if __name__ == "__main__":
    main()
