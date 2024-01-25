"""Compute metrics on sampled images"""
import sys


sys.path.append(".")
from pathlib import Path
from argparse import ArgumentParser
import re
from typing import Tuple, List
import torch as th
from src.utils.net import get_device, Device
from src.model.resnet import load_classifier_t
from src.model.guided_diff.classifier import load_guided_classifier
from src.utils.metrics import accuracy, hard_label_from_logit, prob_vec_from_logit, r3_accuracy, top_n_accuracy
from exp.utils import SimulationConfig


def main():
    args = parse_args()
    # Setup and assign a directory where simulation results are saved.
    classes_and_samples, config = collect_samples(args.res_dir, args.sim_name)
    device = get_device(Device.GPU)

    # Hyper/meta params
    channels, image_size = config.num_channels, config.image_size
    models_dir = Path.cwd() / "models"
    if config.seed is not None:
        print(f"Simulated with fixed seed: {config.seed}")

    if args.metric == "acc" or args.metric == "all":
        # Load classifier
        print("Computing accuracy")

        classifier_path = models_dir / f"{config.classifier}.pt"
        print(config.classifier)
        assert classifier_path.exists(), f"Model '{classifier_path}' does not exist."
        if "mnist" in config.classifier:
            channels, image_size = 1, 28
            batch_size = 10
            classifier = load_classifier_t(model_path=classifier_path, dev=device)
            classifier.eval()
        elif "256x256" in config.classifier:
            channels, image_size = 3, 256
            batch_size = 5
            classifier = load_guided_classifier(model_path=classifier_path, dev=device, image_size=image_size)
            classifier.eval()
        else:
            print(f"Incorrect model '{config.diff_model}'")

        simple_acc, r3_acc = compute_acc(classifier, classes_and_samples, batch_size, device)
        # print(
        #     f"Respaced T: {config.num_respaced_diff_steps}, MCMC stop t<={config.mcmc_lower_t}, Method: {config.mcmc_method}({config.mcmc_steps}), lambda={config.guid_scale}"
        # )
        mcmc_spec = config.mcmc_stepsizes
        factor = mcmc_spec["params"]["factor"]
        exp = mcmc_spec["params"]["exponent"]
        bs = mcmc_spec["beta_schedule"]
        print(
            f"MCMC stop t<={config.mcmc_lower_t}, Method: {config.mcmc_method}({config.mcmc_steps}), lambda={config.guid_scale}, n_trapets: {config.n_trapets}, a={factor}, b={exp}, steps={bs}"
        )
        print(f"Acc: {simple_acc.item():.4f}\nR3 acc: {r3_acc.item():.4f}")
        # compute_top_n_acc(classifier, classes_and_samples[:10], 10, device, 10)


def compute_top_n_acc(classifier, classes_and_samples, batch_size, device, n):
    pred_logits = []
    true_classes = []
    for i, (classes_path, samples_path) in enumerate(classes_and_samples):
        print(f"File {i+1}/{len(classes_and_samples)}")
        samples = th.load(samples_path)
        num_samples = samples.size(0)
        for batch in th.chunk(samples, num_samples // batch_size):
            batch = batch.to(device)
            ts = th.zeros((batch.size(0),)).to(device)
            pred_logits.append(classifier(batch, ts).detach().cpu())

        classes = th.load(classes_path).detach().cpu()
        true_classes.append(classes)
    pred_logits = th.cat(pred_logits, dim=0)
    true_classes = th.cat(true_classes, dim=0)
    acc = top_n_accuracy(pred_logits, true_classes, n)
    print(acc)


def compute_acc(classifier, classes_and_samples, batch_size, device):
    pred_logits = []
    true_classes = []
    for i, (classes_path, samples_path) in enumerate(classes_and_samples):
        print(f"File {i+1}/{len(classes_and_samples)}")
        samples = th.load(samples_path)
        num_samples = samples.size(0)
        for batch in th.chunk(samples, num_samples // batch_size):
            batch = batch.to(device)
            ts = th.zeros((batch.size(0),)).to(device)
            pred_logits.append(classifier(batch, ts).detach().cpu())

        classes = th.load(classes_path).detach().cpu()
        true_classes.append(classes)
    pred_logits = th.cat(pred_logits, dim=0)
    true_classes = th.cat(true_classes, dim=0)
    simple_acc = accuracy(hard_label_from_logit(pred_logits), true_classes)
    r3_acc = r3_accuracy(prob_vec_from_logit(pred_logits), true_classes)
    return simple_acc, r3_acc


PATTERN = re.compile(r".*_(\d+)_(\d+).th")


def collect_samples(res_dir: Path, sim_name: str) -> Tuple[List[Tuple[Path, Path]], SimulationConfig]:
    """Collect sampled images and corresponding classes

    Results are saved to timestamped dirs to prevent overwriting, we need to collect samples from all dirs with matching sim name.
    """
    classes = []
    samples = []
    for sub_dir in res_dir.glob(f"{sim_name}*"):
        print("Checking sub dir", sub_dir)
        config = SimulationConfig.from_json(sub_dir / "config.json")
        # assert config.name in sim_name
        classes.extend(sorted([path.name for path in sub_dir.glob("classes_*_*.th")]))
        classes = [sub_dir / cl for cl in classes]
        samples.extend(sorted([path.name for path in sub_dir.glob("samples_*_*.th")]))
        samples = [sub_dir / sm for sm in samples]
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
    parser.add_argument("--sim_name", type=str, required=True, help="Name of simulation name in config")
    parser.add_argument("--metric", default="all", type=str, choices=["all", "acc", "fid"], help="Metric to compute")
    return parser.parse_args()


if __name__ == "__main__":
    main()
