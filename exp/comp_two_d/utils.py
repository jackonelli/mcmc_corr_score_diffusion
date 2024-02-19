from dataclasses import dataclass, asdict
from pathlib import Path
import json
from copy import deepcopy
from typing import Optional


@dataclass
class SimulationConfig:
    # Models
    diff_model_1: str
    diff_model_2: str
    param: str
    # Diffusion
    num_diff_steps: int
    num_respaced_diff_steps: int
    num_samples: int
    batch_size: int
    # MCMC
    mcmc_method: Optional[str]
    mcmc_steps: Optional[int]
    use_rev: bool
    # Value for disabling MCMC steps:
    mcmc_lower_t: Optional[int]
    # Dict that include all necessary information for step sizes of the MCMC method
    mcmc_stepsize: Optional[float]
    # Number of trapezoidal steps
    n_trapets: Optional[int]
    # Seed
    seed: Optional[int] = None
    # Meta
    save_traj: bool = False
    results_dir: Path = Path.cwd() / "results/comp_two_d"

    @staticmethod
    def from_json(cfg_file_path: Path):
        with open(cfg_file_path) as cfg_file:
            cfg = json.load(cfg_file)
        cfg = SimulationConfig(**cfg)
        cfg.results_dir = Path(cfg.results_dir)
        cfg._validate()
        return cfg

    def get_method(self) -> str:
        return self.mcmc_method if self.mcmc_method is not None else "rev"

    def _validate(self):
        assert self.param in self.diff_model_1
        assert self.param in self.diff_model_2
        if self.mcmc_method is not None:
            assert self.mcmc_stepsize is not None
            assert self.mcmc_steps is not None

    def save(self, sim_dir: Path):
        tmp_config = deepcopy(self)
        tmp_config.results_dir = str(tmp_config.results_dir)
        with open(sim_dir / "config.json", "w") as outfile:
            json.dump(asdict(tmp_config), outfile, indent=4, sort_keys=False)

    def same_exp(self, cfg) -> bool:
        same = True
        # Basics
        same &= self.num_diff_steps == cfg.num_diff_steps
        same &= self.num_respaced_diff_steps == cfg.num_respaced_diff_steps
        same &= self.num_diff_steps == cfg.num_diff_steps
        # Models
        same &= self.diff_model_1 == cfg.diff_model_1
        same &= self.diff_model_2 == cfg.diff_model_2
        # MCMC
        same &= self.mcmc_method == cfg.mcmc_method
        same &= self.mcmc_steps == cfg.mcmc_steps
        same &= self.mcmc_stepsize == cfg.mcmc_stepsize
        same &= self.n_trapets == cfg.n_trapets
        return same


def setup_results_dir(config: SimulationConfig, job_id: Optional[int] = None) -> Path:
    config.results_dir.mkdir(exist_ok=True, parents=True)
    method = config.mcmc_method if config.mcmc_method is not None else "rev"
    id = job_id if job_id is not None else 0
    sim_id = f"{config.param}_{method}_{id}"
    sim_dir = config.results_dir / sim_id
    sim_dir.mkdir(exist_ok=True)
    config.save(sim_dir)
    return sim_dir
