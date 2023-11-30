from dataclasses import dataclass, asdict
from typing import Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
from copy import deepcopy
from typing import Tuple
import pickle
import torch as th


def test():
    cfg = SimulationConfig.from_json(Path.cwd() / "exp/configs/hmc.json")
    cfg.save(Path.cwd() / "results")
    print(cfg.mcmc_bounds, type(cfg.mcmc_bounds))


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
    # accept ratio bounds in percent
    mcmc_bounds: Optional[Tuple[float, float]]
    # Meta
    results_dir: Path = Path.cwd() / "results"

    @staticmethod
    def from_json(cfg_file_path: Path):
        with open(cfg_file_path) as cfg_file:
            cfg = json.load(cfg_file)
        cfg = SimulationConfig(**cfg)
        if cfg.mcmc_bounds is not None:
            cfg.mcmc_bounds = tuple(cfg.mcmc_bounds)
        cfg.results_dir = Path(cfg.results_dir)
        cfg._validate()
        return cfg

    def _validate(self):
        if self.mcmc_method is not None:
            assert self.mcmc_steps is not None and self.mcmc_bounds is not None
            for b in self.mcmc_bounds:
                assert b >= 0 and b <= 100
            assert self.mcmc_bounds[0] < self.mcmc_bounds[1]

    def save(self, sim_dir: Path):
        tmp_config = deepcopy(self)
        tmp_config.results_dir = str(tmp_config.results_dir)
        with open(sim_dir / "config.json", "w") as outfile:
            json.dump(asdict(tmp_config), outfile, indent=4, sort_keys=False)


def get_step_size(step_size_dir: Path, bounds: Tuple[float, float]):
    path = step_size_dir / f"step_size_{bounds[0]}_{bounds[1]}.p"
    assert path.exists(), f"Step size file '{path}' not found"
    with open(path, "rb") as f:
        res = pickle.load(f)
    step_sizes = th.tensor([val["step_sizes"][-1] for val in res.values()])
    return step_sizes


def setup_results_dir(config: SimulationConfig) -> Path:
    assert config.results_dir.exists()
    sim_dir = config.results_dir / f"{config.name}_{_timestamp()}"
    sim_dir.mkdir(exist_ok=True)
    config.save(sim_dir)
    return sim_dir


def _timestamp() -> str:
    return datetime.now().strftime("%y%m%d_%H%M")


if __name__ == "__main__":
    test()
