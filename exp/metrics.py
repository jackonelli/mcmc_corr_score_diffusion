"""Compute metrics on sampled images"""
import sys


sys.path.append(".")
from pathlib import Path
from argparse import ArgumentParser
import re
from typing import Tuple, List
import torch as th
from src.diffusion.base import DiffusionSampler
from src.diffusion.beta_schedules import (
    linear_beta_schedule,
    respaced_beta_schedule,
)
from src.utils.net import get_device, Device
from src.model.guided_diff.unet import NUM_CLASSES, load_guided_diff_unet
from src.model.guided_diff.classifier import load_guided_classifier
from src.guidance.base import GuidanceSampler, MCMCGuidanceSampler
from src.guidance.classifier_full import ClassifierFullGuidance
from src.samplers.mcmc import AnnealedHMCScoreSampler
from src.utils.metrics import accuracy, hard_label_from_logit
from exp.utils import SimulationConfig, setup_results_dir, get_step_size

PATTERN = re.compile(r".*_(\d+)_(\d+).th")


def collect_samples(res_dir: Path, sim_name: str) -> Tuple[List[Tuple[Path, Path]], SimulationConfig]:
    """Collect sampled images and corresponding classes

    Results are saved to timestamped dirs to prevent overwriting, we need to collect samples from all dirs with matching sim name.
    """
    classes = []
    samples = []
    for sub_dir in res_dir.glob(f"{sim_name}_*"):
        print("Checking sub dir", sub_dir)
        config = SimulationConfig.from_json(sub_dir / "config.json")
        assert sim_name == config.name
        classes.extend(sorted([path.name for path in sub_dir.glob("classes_*_*.th")]))
        classes = [sub_dir / cl for cl in classes]
        samples.extend(sorted([path.name for path in sub_dir.glob("samples_*_*.th")]))
        samples = [sub_dir / sm for sm in samples]
    assert len(samples) == len(classes)
    print(f"Found {len(samples)} samples")
    coll = list(zip(classes, samples))
    validate(coll)
    return coll, config


def tmp_samples(res_dir: Path, sim_name: str) -> Tuple[List[Tuple[Path, Path]], SimulationConfig]:
    """Collect sampled images and corresponding classes

    Results are saved to timestamped dirs to prevent overwriting, we need to collect samples from all dirs with matching sim name.
    """
    samples = []
    for sub_dir in res_dir.glob(f"{sim_name}_*"):
        print("Checking sub dir", sub_dir)
        config = SimulationConfig.from_json(sub_dir / "config.json")
        assert sim_name == config.name
        samples.extend(sorted([path.name for path in sub_dir.glob("samples_*_*.th")]))
        samples = [sub_dir / sm for sm in samples]
    coll = list(zip([None] * len(samples), samples))
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


def main():
    args = parse_args()
    # Setup and assign a directory where simulation results are saved.
    # classes_and_samples, config = collect_samples(args.res_dir, args.sim_name)
    classes_and_samples, config = tmp_samples(args.res_dir, args.sim_name)
    device = get_device(Device.GPU)

    # Hyper/meta params
    channels, image_size = config.num_channels, config.image_size
    models_dir = Path.cwd() / "models"
    if args.metric == "acc" or args.metric == "all":
        # Load classifier
        print("Computing accuracy")
        classifier_path = models_dir / f"{config.classifier}.pt"
        assert classifier_path.exists(), f"Model '{classifier_path}' does not exist."
        classifier = load_guided_classifier(model_path=classifier_path, dev=device, image_size=image_size)
        classifier.eval()

        compute_acc(classifier, classes_and_samples, 10, device)


def compute_acc(classifier, classes_and_samples, batch_size, device):
    pred_logits = []
    true_classes = []
    for _, samples_path in classes_and_samples[:1]:
        print("file")
        samples = th.load(samples_path)
        num_samples = samples.size(0)
        for batch in th.chunk(samples, num_samples // batch_size):
            batch = batch.to(device)
            ts = th.zeros((batch.size(0),)).to(device)
            pred_logits.append(classifier(batch, ts).detach().cpu())

        classes = th.ones((num_samples,)).long()
        true_classes.append(classes)
    pred_logits = th.cat(pred_logits, dim=0)
    true_classes = th.cat(true_classes, dim=0)
    acc = accuracy(hard_label_from_logit(pred_logits), true_classes)
    print(acc)


def parse_args():
    parser = ArgumentParser(prog="Sample from diffusion model")
    parser.add_argument("--res_dir", type=Path, required=True, help="Parent dir for all results")
    parser.add_argument("--sim_name", type=str, required=True, help="Name of simulation name in config")
    parser.add_argument("--metric", default="all", type=str, choices=["all", "acc", "fid"], help="Metric to compute")
    return parser.parse_args()


if __name__ == "__main__":
    main()
