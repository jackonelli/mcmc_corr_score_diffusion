"""2D product composition experiments

Compute metrics from samples generated by the 'train_script_product' script
"""
from argparse import ArgumentParser
import pickle
from pathlib import Path
import numpy as np

from src.metrics import (
    gmm_metric,
    wasserstein_metric,
    ll_prod_metric,
)


def main():
    args = parse_args()
    gmm_metrics = dict()
    w2_metrics = dict()
    ll_metrics = dict()
    sample_files = args.samples_path.glob("samples*.p")
    for sample_file in sample_files:
        print(f"Computing metrics for '{sample_file.name}'")
        samples = pickle.load(open(sample_file, "rb"))
        # True data distribution
        target = samples["target"]
        for name, sample in samples.items():
            gmm = gmm_metric(target, sample)
            add_metric(gmm_metrics, name, gmm)
            # add_metric(gmm_metrics, name, 1.0)

            w2 = wasserstein_metric(target, sample)
            add_metric(w2_metrics, name, w2)
            # add_metric(w2_metrics, name, 1.0)

            ll = ll_prod_metric(sample)
            add_metric(ll_metrics, name, ll)

    gmm_stats = stats(gmm_metrics)
    w2_stats = stats(w2_metrics)
    ll_stats = stats(ll_metrics)
    if args.tex:
        print_tex_table(comb_stats(ll_stats, w2_stats, gmm_stats))
    else:
        for metric_name, stats_dict in (
            ("LL", ll_stats),
            ("W_2", w2_stats),
            ("GMM", gmm_stats),
        ):
            print_stats(stats_dict, metric_name)


def print_stats(stats_dict, metric_name):
    """Print stats"""
    print(metric_name)
    for name, stat in stats_dict.items():
        mean, std = stat
        print(f"\t{name}: {mean} +/- {std}")


def comb_stats(ll, w2, gmm):
    """Combine multiple metrics into a common dict."""
    comb = dict()
    for name in ll.keys():
        comb[name] = (ll[name], w2[name], gmm[name])
    return comb


def print_tex_table(stats_dict):
    """Helper function to generate a tex formatted table"""
    print("EBM")
    for name, (ll, w2, gmm) in stats_dict.items():
        if name == "target" or "diff" in name:
            continue
        print(
            f"& {NAME_CONV[name]}\t\t & ${ll[0]:.2f} \\pm {ll[1]:.2f}$ & ${w2[0]:.2f} \\pm {w2[1]:.2f}$ & ${gmm[0]:.3f} \\pm {gmm[1]:.5f}$ \\\\"
        )
    print("Diff")
    for name, (ll, w2, gmm) in stats_dict.items():
        if name == "target" or "ebm" in name:
            continue
        print(
            f"& {NAME_CONV[name]}\t\t & ${ll[0]:.2f} \\pm {ll[1]:.2f}$ & ${w2[0]:.2f} \\pm {w2[1]:.2f}$ & ${gmm[0]:.3f} \\pm {gmm[1]:.5f}$ \\\\"
        )


def stats(metrics):
    """Compute stats"""
    stats_ = dict()
    for name, vals in metrics.items():
        stats_[name] = mean_and_std(vals)
    return stats_


def mean_and_std(vals: list):
    """Compute mean and std from a list of numbers"""
    vals_ = np.array(vals)
    return vals_.mean(), vals_.std()


def add_metric(dict_, key, val):
    """Helper method to add a value to existing dict. key or add key."""
    if dict_.get(key) is None:
        dict_[key] = [val]
    else:
        dict_[key].append(val)


def parse_args():
    """Script args"""
    parser = ArgumentParser(prog="compute_metrics_product")
    parser.add_argument(
        "--samples_path",
        default=None,
        type=Path,
        help="Dir. to load samples files from. The script looks for files with the pattern 'samples*.p'",
    )
    parser.add_argument(
        "--tex", action="store_true", help="If set, prints tex formatted output."
    )
    return parser.parse_args()


NAME_CONV = {
    "ebm_hmc": "HMC",
    "ebm_reverse": "Reverse",
    "ebm_uhmc": "U-HMC",
    "ebm_ula": "U-LA",
    "ebm_mala": "LA",
    "diff_hmc4eff": "HMC-4(eff)",
    "diff_hmc3": "HMC-3",
    "diff_reverse": "Reverse",
    "diff_hmc5": "HMC-5",
    "diff_hmc10": "HMC-10",
    "diff_uhmc": "U-HMC",
    "diff_ula": "U-LA",
    "diff_mala3": "LA-3",
    "diff_mala5": "LA-5",
    "diff_mala10": "LA-10",
}

if __name__ == "__main__":
    main()
