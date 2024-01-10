import numpy as np
import torch as th
from pathlib import Path
import matplotlib.pyplot as plt
from src.utils.net import get_device, Device
from src.model.comp_two_d.classifier import load_classifier
from src.model.comp_two_d.diffusion import ResnetDiffusionModel, ResnetDiffusionModelEnergy
from src.data.comp_2d import GmmRadial, Bar
from src.guidance.base import GuidanceSampler, MCMCGuidanceSampler, GuidanceSamplerAcceptanceComparison
from src.diffusion.base import DiffusionSampler
from src.diffusion.beta_schedules import (
    improved_beta_schedule,
    respaced_beta_schedule,
)
from src.samplers.mcmc import (
    AnnealedHMCScoreSampler,
    AnnealedHMCEnergySampler,
    AnnealedLAScoreSampler,
    AnnealedUHMCScoreSampler,
    AnnealedULAScoreSampler,
)
from src.guidance.classifier_full import ClassifierFullGuidance


def load_diff_model(diff_model_path, T, device, energy=False):
    assert diff_model_path.exists(), f"Model '{diff_model_path}' does not exist."
    if energy:
        diff_model = ResnetDiffusionModelEnergy(num_diff_steps=T)
    else:
        diff_model = ResnetDiffusionModel(num_diff_steps=T)
    diff_model.load_state_dict(th.load(diff_model_path))
    diff_model.to(device)
    # diff_model.eval()
    return diff_model


def main():
    T = 100
    guid_scale = 1.
    num_classes = 8
    device = get_device(Device.GPU)
    models_dir = Path.cwd() / "models"
    diff_model = load_diff_model(models_dir / "gmm.pt", T, device)
    diff_model_energy = load_diff_model(models_dir / "gmm_energy.pt", T, device, True)
    classifier = load_classifier(models_dir / "class_t_gmm.pt", num_classes, device, num_diff_steps=T)
    guidance = ClassifierFullGuidance(classifier, lambda_=guid_scale)

    betas, time_steps = respaced_beta_schedule(
        original_betas=improved_beta_schedule(num_timesteps=T),
        T=T,
        respaced_T=T,
    )
    diff_proc = DiffusionSampler(betas, time_steps, posterior_variance="beta")

    mcmc_steps = 10
    step_sizes = {int(t): 0.03 for t in range(0, T)}
    damping_coeff = 0.5
    leapfrog_steps = 3
    mcmc_sampler = AnnealedHMCScoreSampler(
        mcmc_steps, step_sizes, damping_coeff, th.ones_like(betas), leapfrog_steps, guidance.grad
    )

    mcmc_sampler_energy = AnnealedHMCEnergySampler(
        mcmc_steps, step_sizes, damping_coeff, th.ones_like(betas), leapfrog_steps, guidance.grad
    )

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

    guidance_samp = GuidanceSamplerAcceptanceComparison([guid_sampler_energy, guid_sampler])

    th.manual_seed(0)

    num_samples = 1000
    x_dim = 2
    classes = th.randint(low=0, high=num_classes, size=(num_samples,)).long().to(device)
    _, accept_rate = guidance_samp.sample(num_samples, classes, device, th.Size((x_dim,)))


if __name__ == '__main__':
    main()
