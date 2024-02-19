from argparse import ArgumentParser
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from datetime import datetime
from copy import deepcopy
import re
import shutil
from itertools import tee
from typing import Union, Iterator, Tuple, Optional
import pickle
import torch as th


@dataclass
class SimulationConfig:
    name: str
    # Domain
    image_size: int
    num_channels: int
    # Diffusion
    diff_model: str
    class_cond: bool
    num_diff_steps: int
    num_respaced_diff_steps: int
    num_samples: int
    batch_size: int
    # Guidance
    classifier: str
    guid_scale: float
    # MCMC
    mcmc_method: Optional[str]
    mcmc_steps: Optional[int]
    # Value for disabling MCMC steps:
    mcmc_lower_t: Optional[int]
    # Dict that include all necessary information for step sizes of the MCMC method
    mcmc_stepsizes: Optional[dict]
    # Number of trapezoidal steps
    n_trapets: Optional[int]
    # Seed
    seed: Optional[int] = None
    # Meta
    save_traj: bool = False
    results_dir: Path = Path.cwd() / "results"

    @staticmethod
    def from_json(cfg_file_path: Path):
        with open(cfg_file_path) as cfg_file:
            cfg = json.load(cfg_file)
        cfg = SimulationConfig(**cfg)
        cfg.results_dir = Path(cfg.results_dir)
        cfg._validate()
        return cfg

    def _validate(self):
        if self.mcmc_method is not None:
            assert self.mcmc_stepsizes is not None
            assert isinstance(self.mcmc_stepsizes["load"], bool)
            if self.mcmc_stepsizes["load"]:
                assert self.mcmc_stepsizes["bounds"] is not None
            else:
                assert isinstance(self.mcmc_stepsizes["params"]["factor"], float) or isinstance(
                    self.mcmc_stepsizes["params"]["factor"], int
                )
                assert isinstance(self.mcmc_stepsizes["params"]["exponent"], float) or isinstance(
                    self.mcmc_stepsizes["params"]["exponent"], int
                )
                assert self.mcmc_stepsizes["beta_schedule"] is not None
            if self.mcmc_method == "la":
                assert isinstance(self.n_trapets, int)

    def save(self, sim_dir: Path):
        tmp_config = deepcopy(self)
        tmp_config.results_dir = str(tmp_config.results_dir)
        with open(sim_dir / "config.json", "w") as outfile:
            json.dump(asdict(tmp_config), outfile, indent=4, sort_keys=False)

    def same_exp(self, cfg) -> bool:
        same = True
        # Basics
        same &= self.image_size == cfg.image_size
        same &= self.num_channels == cfg.num_channels
        same &= self.num_diff_steps == cfg.num_diff_steps
        same &= self.num_respaced_diff_steps == cfg.num_respaced_diff_steps
        same &= self.num_diff_steps == cfg.num_diff_steps
        # Models
        same &= self.diff_model == cfg.diff_model
        same &= self.classifier == cfg.classifier
        # Guidance
        same &= self.guid_scale == cfg.guid_scale
        same &= self.mcmc_method == cfg.mcmc_method
        same &= self.mcmc_steps == cfg.mcmc_steps
        same &= self.mcmc_stepsizes == cfg.mcmc_stepsizes
        same &= self.n_trapets == cfg.n_trapets
        return same


@dataclass
class UnguidedSimulationConfig:
    name: str
    # Domain
    image_size: int
    num_channels: int
    # Diffusion
    diff_model: str
    class_cond: bool
    num_diff_steps: int
    num_respaced_diff_steps: int
    num_samples: int
    batch_size: int
    # Seed
    seed: Optional[int] = None
    # Meta
    save_traj: bool = False
    results_dir: Path = Path.cwd() / "results"

    @staticmethod
    def from_json(cfg_file_path: Path):
        with open(cfg_file_path) as cfg_file:
            cfg = json.load(cfg_file)
        cfg = UnguidedSimulationConfig(**cfg)
        cfg.results_dir = Path(cfg.results_dir)
        return cfg

    def save(self, sim_dir: Path):
        tmp_config = deepcopy(self)
        tmp_config.results_dir = str(tmp_config.results_dir)
        with open(sim_dir / "config.json", "w") as outfile:
            json.dump(asdict(tmp_config), outfile, indent=4, sort_keys=False)

    def same_exp(self, cfg) -> bool:
        same = True
        same &= self.image_size == cfg.image_size
        same &= self.num_channels == cfg.num_channels
        same &= self.diff_model == cfg.diff_model
        same &= self.num_diff_steps == cfg.num_diff_steps
        same &= self.num_respaced_diff_steps == cfg.num_respaced_diff_steps
        same &= self.num_diff_steps == cfg.num_diff_steps
        return same


def get_step_size(step_size_dir: Path, dataset_name: str, mcmc_method: str, mcmc_accept_bounds: str):
    # print("Warning: using steps from T_resp = 500")
    # steps = 500
    path = step_size_dir / f"{dataset_name}_{mcmc_method}_{mcmc_accept_bounds}.p"
    assert path.exists(), f"Step size file '{path}' not found"
    with open(path, "rb") as f:
        res = pickle.load(f)
    # We accidentally save the last index (which we then leave with a reverse step)
    # Therefore we include t=T in the dict, but it's not populated with a step size.
    extracted = [(int(t), x["step_sizes"][-1]) for t, x in res.items() if x["step_sizes"]]
    return dict(extracted)


def setup_results_dir(config: Union[SimulationConfig, UnguidedSimulationConfig], job_id: Optional[int]) -> Path:
    config.results_dir.mkdir(exist_ok=True, parents=True)
    if job_id is None:
        sim_id = f"{config.name}_{timestamp()}"
    else:
        sim_id = f"{config.name}_{job_id}"
    sim_dir = config.results_dir / sim_id
    sim_dir.mkdir(exist_ok=True)
    config.save(sim_dir)
    return sim_dir


def timestamp() -> str:
    return datetime.now().strftime("%y%m%d_%H%M")


SIM_BATCH_PAT = re.compile(r"samples_(\d+)_(\d+).th")


def compare_configs(srces: Iterator[Path]) -> bool:
    cfgs = map(lambda x: SimulationConfig.from_json(x / "config.json"), srces)
    ref_cfg = next(cfgs)
    same_cfgs = map(lambda x: ref_cfg.same_exp(x), cfgs)
    return all(same_cfgs)


def find_sim_numbers(path: Path) -> Tuple[int, int]:
    m = SIM_BATCH_PAT.match(path.name)
    assert m is not None, f"Unable to parse sim and batch id from '{path.name}'"
    sim, batch = int(m[1]), int(m[2])
    return sim, batch


def copy_files(src_dir: Path, target: Path, max_sim_num: int):
    for ss in src_dir.glob("samples_*_*.th"):
        sim, batch = find_sim_numbers(ss)
        new_sim = max_sim_num + sim
        src = src_dir / f"samples_{sim}_{batch}.th"
        dest = target / f"samples_{new_sim}_{batch}.th"
        shutil.copy(src, dest)
        src = src_dir / f"classes_{sim}_{batch}.th"
        dest = target / f"classes_{new_sim}_{batch}.th"
        shutil.copy(src, dest)


def count_samples(target_dir: Path):
    existing_samples = target_dir.glob("samples_*_*.th")
    return sum(map(lambda x: th.load(x).size(0), existing_samples))


def combine_exps(srces: Iterator[Path], target_dir: Path):
    dir_check, srces = tee(srces)
    any_lacks_config = next(filter(lambda x: not (x / "config.json").exists(), dir_check), None)
    assert any_lacks_config is None, f"'{any_lacks_config.name}' does not have a config.json file"
    cfg_check, srces = tee(srces)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)
    if not compare_configs(cfg_check):
        print("Mismatched config")
        return
    first = next(srces)
    shutil.copy(first / "config.json", target_dir / "config.json")
    copy_files(first, target_dir, 0)
    for src in srces:
        existing_samples = target_dir.glob("samples_*_*.th")
        max_sim_num = max(map(lambda x: find_sim_numbers(x)[0], existing_samples))
        copy_files(src, target_dir, max_sim_num)
    print(f"Created combined dir '{target_dir.name}' with {count_samples(target_dir)} samples")


def main():
    args = parse_args()
    dir_list = map(lambda x: args.parent / Path(x) if args.parent is not None else Path(x), args.dir_list.split(","))
    combine_exps(dir_list, args.dest)


def parse_args():
    parser = ArgumentParser(prog="Exp utils CLI")
    subparsers = parser.add_subparsers(help="sub parser")

    combine_parser = subparsers.add_parser("combine", help="Combine samples with the same exp config")
    combine_parser.add_argument("--dir_list", type=str, help="comma separated list of dirs to combine", required=True)
    combine_parser.add_argument(
        "--parent",
        type=str,
        help="optional parent directory, which dirs in dir_list are relative to, if None, dir_list paths are treated as absolute",
    )
    combine_parser.add_argument("--dest", type=Path, help="output dir", required=True)

    return parser.parse_args()


if __name__ == "__main__":
    main()
