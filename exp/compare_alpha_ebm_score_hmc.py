import os
import sys

sys.path.append(".")
from pathlib import Path
from argparse import ArgumentParser
from exp.sample_guided_diff import load_models, get_guid_sampler
from src.utils.net import get_device, Device
import scipy
import pickle
from exp.utils import SimulationConfig, setup_results_dir
from src.utils.seeding import set_seed
from src.diffusion.beta_schedules import (
    respaced_beta_schedule,
)
from src.diffusion.base import DiffusionSampler
from src.guidance.classifier_full import ClassifierFullGuidance
import torch as th
from src.guidance.base import reverse_func_require_grad, reverse_func
from src.samplers.mcmc import (get_v_prime, leapfrog_steps, transition_hmc,
                               estimate_energy_diff_linear_given,
                               estimate_energy_diff_linear_given_require_grad)


def main():
    args = parse_args()
    device = get_device(Device.GPU)

    config_score = SimulationConfig.from_json(args.score_config)
    config_energy = SimulationConfig.from_json(args.energy_config)

    set_seed(config_score.seed)

    (diff_model_score, classifier, dataset, beta_schedule, post_var, _) = load_models(config_score, device)
    (diff_model_energy, _, _, _, _, _) = load_models(config_energy, device)

    sim_dir = setup_results_dir(config_score, args.job_id)
    _ = setup_results_dir(config_energy, args.job_id, suffix="_energy")
    dataset_name, image_size, num_classes, num_channels = dataset

    betas, time_steps = respaced_beta_schedule(
        original_betas=beta_schedule(num_timesteps=config_score.num_diff_steps),
        T=config_score.num_diff_steps,
        respaced_T=config_score.num_respaced_diff_steps,
    )

    diff_sampler = DiffusionSampler(betas, time_steps, posterior_variance=post_var)
    guidance = ClassifierFullGuidance(classifier, lambda_=config_score.guid_scale)

    guid_sampler_score = get_guid_sampler(config_score, diff_model_score, diff_sampler, guidance, time_steps,
                                          'cifar100', False)
    guid_sampler_energy = get_guid_sampler(config_energy, diff_model_energy, diff_sampler, guidance, time_steps,
                                           'cifar100', True)
    # Sample
    num_samples = config_score.num_samples
    max_n_trapz = args.max_n_trapz

    shape = th.Size((num_channels, image_size, image_size))
    classes = th.randint(low=0, high=num_classes, size=(config_energy.batch_size,)).long().to(device)
    th.random.manual_seed(0)
    x_tm1 = th.randn((num_samples,) + shape).to(device)

    differences = {}
    for t, t_idx in zip(diff_sampler.time_steps.__reversed__(), reversed(diff_sampler.time_steps_idx)):
        print(t_idx)
        if args.follow_score:
            x_tm1 = reverse_func(guid_sampler_score, t, t_idx, x_tm1, classes, device, False).detach()
        else:
            x_tm1 = reverse_func_require_grad(guid_sampler_energy, t, t_idx, x_tm1, classes, device, False).detach()
        respaced_t = diff_sampler.time_steps[t_idx - 1].item()
        x = x_tm1
        self = guid_sampler_energy.mcmc_sampler
        if guid_sampler_energy._mcmc_sampling_predicate(respaced_t) and t_idx > 0:
            t = respaced_t
            differences[t] = {i: {} for i in range(self.num_samples_per_step)}
            t_idx = t_idx - 1
            dims = x.dim()
            v = th.randn_like(x) * self._mass_diag_sqrt[t_idx]
            for i in range(self.num_samples_per_step):
                v_prime = get_v_prime(v=v, damping_coeff=self._damping_coeff,
                                      mass_diag_sqrt=self._mass_diag_sqrt[t_idx])
                x_next, v_next, _, _ = leapfrog_steps(
                    x_0=x,
                    v_0=v_prime,
                    t=t,
                    t_idx=t_idx,
                    gradient_function=self.gradient_function,
                    step_size=self.step_sizes[t],
                    mass_diag_sqrt=self._mass_diag_sqrt[t_idx],
                    num_steps=self._num_leapfrog_steps,
                    classes=classes,
                )
                x_next = x_next.detach()
                logp_v_p, logp_v = transition_hmc(
                    v_prime=v_prime, v_next=v_next, mass_diag_sqrt=self._mass_diag_sqrt[t_idx], dims=dims
                )
                energy_diff = (self.energy_function(x_next, t, t_idx, classes).detach()
                               - self.energy_function(x, t, t_idx, classes).detach())
                logp_accept = logp_v - logp_v_p + energy_diff

                alpha = th.exp(logp_accept)
                alpha = th.clip(alpha, 0., 1.)
                differences[t][i]['energy_alpha'] = alpha.cpu()
                # print(f'alpha: {alpha.cpu()}')

                self2 = guid_sampler_score.mcmc_sampler

                classifier_energy_diff = (self2.class_log_prob(x_next, t, t_idx, classes) -
                                          self2.class_log_prob(x, t, t_idx, classes))

                n_trapets = 5
                differences[t][i]['alpha_diff'] = []
                differences[t][i]['n_trapets'] = []
                differences[t][i]['energy_alpha_approx'] = []
                differences[t][i]['score_alpha'] = []
                # differences[t][i]['spearmanr'] = []
                # differences[t][i]['pearsonr'] = []

                alpha_approx = None
                difference_score_ebm = th.inf
                while n_trapets <= max_n_trapz and difference_score_ebm > 0.05:
                    grads = [None for _ in range(n_trapets)]
                    intermediate_steps = th.linspace(0, 1, steps=n_trapets).to(x.device)
                    energy_diff_approx = estimate_energy_diff_linear_given(
                        self2.grad_diff, grads, x, x_next, t, t_idx, intermediate_steps, classes, dims
                    )
                    energy_diff_approx += classifier_energy_diff

                    logp_accept_approx = logp_v - logp_v_p + energy_diff_approx
                    alpha_approx = th.exp(logp_accept_approx)

                    clip_alpha_approx = th.clip(alpha_approx, 0., 1.)

                    grads = [None for _ in range(n_trapets)]
                    energy_diff_approx_ebm = estimate_energy_diff_linear_given_require_grad(
                        self.grad_diff, grads, x, x_next, t, t_idx, intermediate_steps, classes, dims
                    )
                    energy_diff_approx_ebm += classifier_energy_diff

                    logp_accept_approx_ebm = logp_v - logp_v_p + energy_diff_approx_ebm
                    alpha_approx_ebm = th.exp(logp_accept_approx_ebm)

                    clip_alpha_approx_ebm = th.clip(alpha_approx_ebm, 0., 1.)

                    difference_score_ebm = (clip_alpha_approx - alpha).abs().mean().item()
                    differences[t][i]['score_alpha'].append(clip_alpha_approx.cpu())
                    differences[t][i]['energy_alpha_approx'].append(clip_alpha_approx_ebm.cpu())
                    differences[t][i]['alpha_diff'].append(difference_score_ebm)
                    differences[t][i]['n_trapets'].append(n_trapets)
                    # differences[t][i]['pearsonr'].append(scipy.stats.pearsonr(clip_alpha_approx.cpu(), alpha.cpu())[0])
                    # differences[t][i]['spearmanr'].append(scipy.stats.spearmanr(clip_alpha_approx.cpu(), alpha.cpu())[0])

                    # print(f'score alpha: {clip_alpha_approx.cpu()}')
                    # print(f'energy alpha approx: {clip_alpha_approx_ebm.cpu()}')
                    # print(f'pearsonr: {scipy.stats.pearsonr(clip_alpha_approx.cpu(), alpha.cpu())[0]}')
                    # print(f'spearmanr: {scipy.stats.spearmanr(clip_alpha_approx.cpu(), alpha.cpu())[0]}')
                    # print(f'difference_score_ebm: {difference_score_ebm}')
                    n_trapets += 5

                u = th.rand(x_next.shape[0]).to(x_next.device)
                if args.follow_score:
                    follow_alpha = alpha_approx
                else:
                    follow_alpha = alpha

                accept = (u < follow_alpha).to(th.float32).reshape((x.shape[0],) + tuple(([1 for _ in range(dims - 1)])))
                x = accept * x_next + (1 - accept) * x
                v = accept * v_next + (1 - accept) * v_prime

        x_tm1 = x

    th.save(x_tm1, sim_dir / f"samples_{args.sim_batch}_{0}.th")
    th.save(classes, sim_dir / f"classes_{args.sim_batch}_{0}.th")
    pickle.dump(differences, open(sim_dir / f"alpha_differences.p", 'wb'))


def parse_args():
    parser = ArgumentParser(prog="Sample from diffusion model and compare alpha between energy and score")
    parser.add_argument("--score_config", type=Path, required=True, help="Score config file path")
    parser.add_argument("--energy_config", type=Path, required=True, help="Energy config file path")
    parser.add_argument("--follow_score", action='store_true', help="Follow score model otherwise following energy")
    parser.add_argument("--max_n_trapz", type=int, default=50, help="Maximum number of trapets steps")
    parser.add_argument(
        "--job_id", type=int, default=None, help="Simulation batch index, indexes parallell simulations."
    )
    parser.add_argument(
        "--sim_batch", type=int, default=0, help="Simulation batch index, indexes parallell simulations."
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()

