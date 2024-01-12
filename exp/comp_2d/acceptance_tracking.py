import torch as th
from pathlib import Path
from src.utils.net import get_device, Device
from src.model.comp_two_d.classifier import load_classifier
from src.guidance.base import GuidanceSampler, MCMCGuidanceSampler, GuidanceSamplerAcceptanceComparison
from src.diffusion.base import DiffusionSampler
from src.diffusion.beta_schedules import (
    improved_beta_schedule,
    respaced_beta_schedule,
)
from src.samplers.mcmc import (
    AnnealedHMCScoreSampler,
    AnnealedHMCEnergySampler,
    AnnealedLAEnergySampler,
    AnnealedLAScoreSampler,
    AnnealedUHMCScoreSampler,
    AnnealedULAScoreSampler,
)
from src.guidance.classifier_full import ClassifierFullGuidance
from src.model.comp_two_d.diffusion import load_diff_model_gmm
from datetime import datetime
import pickle


def main():
    T = 100
    guid_scale = 1.
    num_classes = 8
    device = get_device(Device.GPU)
    models_dir = Path.cwd() / "models"
    diff_model = load_diff_model_gmm(models_dir / "gmm.pt", T, device)
    diff_model_energy = load_diff_model_gmm(models_dir / "gmm_energy.pt", T, device, True)
    classifier = load_classifier(models_dir / "class_t_gmm.pt", num_classes, device, num_diff_steps=T)
    guidance = ClassifierFullGuidance(classifier, lambda_=guid_scale)

    betas, time_steps = respaced_beta_schedule(
        original_betas=improved_beta_schedule(num_timesteps=T),
        T=T,
        respaced_T=T,
    )
    diff_proc = DiffusionSampler(betas, time_steps, posterior_variance="beta")

    mcmc_steps = 1
    step_sizes = {int(t): 0.001 for t in range(0, T)}

    mcmc_sampler_energy = AnnealedLAEnergySampler(mcmc_steps, step_sizes, guidance.grad)
    mcmc_sampler = AnnealedLAScoreSampler(mcmc_steps, step_sizes, guidance.grad)

    guid_sampler = MCMCGuidanceSampler(
        diff_model=diff_model,
        diff_proc=diff_proc,
        guidance=guidance,
        mcmc_sampler=mcmc_sampler,
        reverse=True,
        diff_cond=False,
    )

    guid_sampler_energy = MCMCGuidanceSampler(
        diff_model=diff_model_energy,
        diff_proc=diff_proc,
        guidance=guidance,
        mcmc_sampler=mcmc_sampler_energy,
        reverse=True,
        diff_cond=False,
    )

    samplers = [guid_sampler_energy, guid_sampler]
    guidance_samp = GuidanceSamplerAcceptanceComparison(samplers)

    th.manual_seed(0)

    num_samples = 1000
    x_dim = 2
    classes = th.randint(low=0, high=num_classes, size=(num_samples,)).long().to(device)
    accept_rate, energy_diff = guidance_samp.accept_ratio_one_guides_LA(num_samples, classes, device, th.Size((x_dim,)))

    res_dir = Path.cwd() / "results"
    assert res_dir.exists()
    sim_dir = res_dir / f"acceptance_tracking_comp2d_{timestamp()}"
    sim_dir.mkdir(exist_ok=True)
    save_path = sim_dir / f"num_saplers_{len(samplers)}_LA.p"
    res = {'accept_rate': accept_rate, 'energy_diff': energy_diff}
    pickle.dump(res, open(save_path, "wb"))


def timestamp() -> str:
    return datetime.now().strftime("%y%m%d_%H%M")


if __name__ == '__main__':
    main()
