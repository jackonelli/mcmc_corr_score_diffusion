from src.diffusion.beta_schedules import (
    improved_beta_schedule,
    linear_beta_schedule,
    respaced_beta_schedule,
)
from src.guidance.base import GuidanceSampler, MCMCGuidanceSampler
from src.samplers.mcmc import (
    AnnealedHMCEnergySampler,
    AnnealedHMCEnergyApproxSampler,
    AnnealedHMCScoreSampler,
    AnnealedHMCScoreNumberTrapsSampler,
    AnnealedUHMCEnergySampler,
    AnnealedUHMCScoreSampler,
    AnnealedULAScoreSampler,
    AnnealedULAEnergySampler,
    AnnealedLAScoreSampler,
    AnnealedLAEnergySampler,
)
from exp.utils import get_step_size



def get_guid_sampler(config, diff_model, diff_sampler, guidance, time_steps, dataset_name, energy_param: bool,
                     MODELS_DIR, save_grad=False):
    if config.mcmc_method is None:
        guid_sampler = GuidanceSampler(diff_model, diff_sampler, guidance, diff_cond=config.class_cond,
                                       save_grad=save_grad)
    else:
        assert config.mcmc_steps is not None
        assert config.mcmc_method is not None
        assert config.mcmc_stepsizes is not None
        if config.mcmc_stepsizes["load"]:
            print("Load step sizes for MCMC.")
            step_sizes = get_step_size(
                MODELS_DIR / "step_sizes", dataset_name, config.mcmc_method, config.mcmc_stepsizes["bounds"],
                str(config.num_diff_steps)
            )
        else:
            print("Use parameterized step sizes for MCMC.")
            if config.mcmc_stepsizes["beta_schedule"] == "lin":
                beta_schedule_mcmc = linear_beta_schedule
            elif config.mcmc_stepsizes["beta_schedule"] == "cos":
                beta_schedule_mcmc = improved_beta_schedule
            else:
                print("mcmc_stepsizes.beta_schedule must be 'lin' or 'cos'.")
                raise ValueError
            betas_mcmc, _ = respaced_beta_schedule(
                original_betas=beta_schedule_mcmc(num_timesteps=config.num_diff_steps),
                T=config.num_diff_steps,
                respaced_T=config.num_respaced_diff_steps,
            )
            a = config.mcmc_stepsizes["params"]["factor"]
            b = config.mcmc_stepsizes["params"]["exponent"]
            step_sizes = {int(t.item()): a * beta**b for (t, beta) in zip(time_steps, betas_mcmc)}

        if config.mcmc_method == "hmc":
            if energy_param:
                if config.n_trapets < 0:
                    mcmc_sampler = AnnealedHMCEnergySampler(config.mcmc_steps, step_sizes, 0.9,
                                                            diff_sampler.betas, 3, None)
                else:
                    mcmc_sampler = AnnealedHMCEnergyApproxSampler(config.mcmc_steps, step_sizes, 0.9,
                                                              diff_sampler.betas, 3, None,
                                                              n_intermediate_steps=config.n_trapets, exact_energy=False)
            else:
                """
                mcmc_sampler = AnnealedHMCScoreSampler(config.mcmc_steps, step_sizes,
                                                       0.9, diff_sampler.betas, 3,
                                                       None, n_intermediate_steps=config.n_trapets)
                """
                mcmc_sampler = AnnealedHMCScoreNumberTrapsSampler(config.mcmc_steps, step_sizes,
                                                       0.9, diff_sampler.betas, 3,
                                                       None, n_intermediate_steps=config.n_trapets)

        elif config.mcmc_method == "la":
            assert config.n_trapets is not None
            if energy_param:
                mcmc_sampler = AnnealedLAEnergySampler(config.mcmc_steps, step_sizes, None)
            else:
                mcmc_sampler = AnnealedLAScoreSampler(config.mcmc_steps, step_sizes, None, n_trapets=config.n_trapets)
        elif config.mcmc_method == "uhmc":
            if energy_param:
                mcmc_sampler = AnnealedUHMCEnergySampler(
                    config.mcmc_steps, step_sizes, 0.9, diff_sampler.betas, 3, None
                )
            else:
                mcmc_sampler = AnnealedUHMCScoreSampler(config.mcmc_steps, step_sizes, 0.9, diff_sampler.betas, 3, None)
        elif config.mcmc_method == "ula":
            if energy_param:
                mcmc_sampler = AnnealedULAEnergySampler(config.mcmc_steps, step_sizes, None)
            else:
                mcmc_sampler = AnnealedULAScoreSampler(config.mcmc_steps, step_sizes, None)
        else:
            raise ValueError(f"Incorrect MCMC method: '{config.mcmc_method}'")

        guid_sampler = MCMCGuidanceSampler(
            diff_model=diff_model,
            diff_proc=diff_sampler,
            guidance=guidance,
            mcmc_sampler=mcmc_sampler,
            reverse=True,
            diff_cond=config.class_cond,
            save_grad=save_grad,
        )
    return guid_sampler
