from dataclasses import dataclass, asdict
from typing import Optional, Union
from pathlib import Path
import json
from datetime import datetime
from copy import deepcopy
import pickle



def test():
    cfg = SimulationConfig.from_json(Path.cwd() / "exp/configs/cifar100_guided_score_reverse.json")
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
    t_skip: int = 0

    @staticmethod
    def load(cfg_file_path: Path):
        with open(cfg_file_path) as cfg_file:
            cfg = json.load(cfg_file)
        return cfg


    @staticmethod
    def from_json_no_load(cfg):
        cfg = SimulationConfig(**cfg)
        cfg.results_dir = Path(cfg.results_dir)
        cfg._validate()
        return cfg

    @staticmethod
    def from_json(cfg_file_path: Path):
        cfg = SimulationConfig.load(cfg_file_path)
        cfg = SimulationConfig.from_json_no_load(cfg)
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

    def save(self, sim_dir: Path, suffix = ""):
        tmp_config = deepcopy(self)
        tmp_config.results_dir = str(tmp_config.results_dir)
        with open(sim_dir / f"config{suffix}.json", "w") as outfile:
            json.dump(asdict(tmp_config), outfile, indent=4, sort_keys=False)


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
        cfg = UnguidedSimulationConfig.load(cfg_file_path)
        cfg = UnguidedSimulationConfig.from_json_no_load(cfg)
        return cfg

    @staticmethod
    def load(cfg_file_path: Path):
        with open(cfg_file_path) as cfg_file:
            cfg = json.load(cfg_file)
        return cfg

    @staticmethod
    def from_json_no_load(cfg):
        cfg = UnguidedSimulationConfig(**cfg)
        cfg.results_dir = Path(cfg.results_dir)
        return cfg

    def save(self, sim_dir: Path):
        tmp_config = deepcopy(self)
        tmp_config.results_dir = str(tmp_config.results_dir)
        with open(sim_dir / "config.json", "w") as outfile:
            json.dump(asdict(tmp_config), outfile, indent=4, sort_keys=False)


def get_step_size(step_size_dir: Path, dataset_name: str, mcmc_method: str, mcmc_accept_bounds: str, num_steps: str):
    path = step_size_dir / f"{mcmc_method}_{dataset_name}_{num_steps}_{mcmc_accept_bounds}.p"
    assert path.exists(), f"Step size file '{path}' not found"
    with open(path, "rb") as f:
        res = pickle.load(f)

    step_size = {k: v for k, v in zip([i for i in range(len(res['best']['step_sizes']))], res['best']['step_sizes'])}
    return step_size


def setup_results_dir(config: Union[SimulationConfig, UnguidedSimulationConfig], job_id: Optional[int],
                      suffix: str = "") -> Path:
    config.results_dir.mkdir(exist_ok=True, parents=True)
    if job_id is None:
        sim_id = f"{config.name}_{timestamp()}"
    else:
        sim_id = f"{config.name}_{job_id}"
    sim_dir = config.results_dir / sim_id
    sim_dir.mkdir(exist_ok=True)
    config.save(sim_dir, suffix)
    return sim_dir


def timestamp() -> str:
    return datetime.now().strftime("%y%m%d_%H%M")


if __name__ == "__main__":
    test()
