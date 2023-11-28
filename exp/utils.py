from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path
import json
from datetime import datetime
from copy import deepcopy


def test():
    cfg = SimulationConfig.from_json(Path.cwd() / "exp/configs/baseline.json")
    cfg.save(Path.cwd() / "results")
    print(timestamp())


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
    classifier: Optional[Path]
    guid_scale: Optional[float]
    # MCMC
    mcmc_steps: Optional[int]
    # Meta
    results_dir: Path = Path.cwd() / "results"

    @staticmethod
    def from_json(cfg_file_path: Path):
        with open(cfg_file_path) as cfg_file:
            cfg = json.load(cfg_file)
        return SimulationConfig(**cfg)

    def save(self, sim_dir: Path):
        tmp_config = deepcopy(self)
        tmp_config.results_dir = str(tmp_config.results_dir)
        with open(sim_dir / "config.json", "w") as outfile:
            json.dump(asdict(tmp_config), outfile, indent=4, sort_keys=False)


def timestamp() -> str:
    return datetime.now().strftime("%y%m%d_%H%M%S")


if __name__ == "__main__":
    test()
