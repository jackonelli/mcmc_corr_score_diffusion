"""
Code is based on https://github.com/yilundu/reduce_reuse_recycle commit 513361e60bb677dec75c086a234715f3db97ea51
"""
import sys

sys.path.append(".")

from argparse import ArgumentParser
import matplotlib.pyplot as plt
from src.utils.tapestry_components import visualize_context, IFPipeline, context_examples
from diffusers import DiffusionPipeline
from src.samplers.mcmc import AnnealedLAScoreSampler, AnnealedULAScoreSampler, MCMCMHCorrSampler
import torch
import pickle
import numpy as np
import json
import os
from src.utils.seeding import set_seed
from pathlib import Path
from datetime import datetime


has_cuda = torch.cuda.is_available()
device = torch.device('cpu' if not has_cuda else 'cuda')
print(f'torch device {device}')


def plot_image(latents_):
    image = latents_[0].cpu().numpy().transpose(1, 2, 0)
    image = ((image + 1) / 2 * 255).astype(np.uint8)
    plt.imshow(image)


def parse_args():
    parser = ArgumentParser(prog="Sample a tapestry image")
    parser.add_argument("--config", type=Path, required=True, help="Config file path")
    parser.add_argument(
        "--job_id", type=int, default=None, help="Simulation batch index, indexes parallell simulations."
    )
    parser.add_argument(
        "--sim_batch", type=int, default=0, help="Simulation batch index, indexes parallell simulations."
    )
    return parser.parse_args()


if __name__ == '__main__':

    print('Set up experiment')
    # Tunable Parameters
    args = parse_args()
    config_path = args.config
    with open(config_path) as cfg_file:
        config = json.load(cfg_file)

    # initialize model
    print('Load model')
    if config['cache'] is not None:
        custom_cache_dir = os.path.expanduser(config['cache'])
    else:
        custom_cache_dir = None
    stage_1 = IFPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16,
                                         use_auth_token=True, cache_dir=custom_cache_dir)
    stage_1.enable_xformers_memory_efficient_attention()
    stage_1.enable_model_cpu_offload()
    stage_1.safety_checker = None
    print('Successfully loaded the model')

    res_dir = Path(config['results_dir'])
    res_dir.mkdir(exist_ok=True, parents=True)
    res_dir = res_dir / 'tapestry'
    res_dir.mkdir(exist_ok=True)

    # Set Seed
    seed = config['seed']
    generator = torch.Generator('cuda')
    if seed is not None:
        generator.manual_seed(seed)
        set_seed(seed)

    # Guidance Magnitude
    guidance_mag = config['guidance_mag']
    guidance_scale = config['guidance_scale']
    idx = config['context_index']

    context = context_examples(idx, guidance_mag)

    # Increase the number of MCMC steps run to sample between intermediate distributions
    mcmc_steps = config['mcmc_steps']

    # Steps sizes as a function of beta
    a = config['parameters']['a']
    b = config['parameters']['b']
    step_sizes = float(a) * stage_1.scheduler.betas ** float(b)

    # Number of reverse steps
    steps = config['n_steps']

    # Stage 2
    stage2 = False

    sampler_text = 'reverse'

    # Construct Sampler
    if config['mh']:
        sampler = AnnealedLAScoreSampler(mcmc_steps, step_sizes, None, config['n_trapets'])
        sampler_text = 'LA'
    else:
        sampler = AnnealedULAScoreSampler(mcmc_steps, step_sizes, None)
        sampler_text = 'ULA'

    # Save image
    time = datetime.now()
    time = time.strftime("%Y%m-%H%M")
    save_file = ('space_seed' + str(config['seed']) + '_' + str(steps) + '_' +
                 sampler_text + '_idx' + str(idx) + '_a_' + config['parameters']['a'] + '_b_'
                 + config['parameters']['b'] + '_traps' + str(config['n_trapets']) + time + '_'
                 + str(args.sim_batch) + '.p')

    color_lookup = {}

    for k, v in context.items():
        color_lookup[v['string']] = (np.random.uniform(size=(3,)), k[0]**2)

    plt.figure(figsize=(5, 5))
    img = visualize_context(128, 64, context, color_lookup)
    plt.imshow(img)
    plt.show()

    for k, v in context.items():
        scale, xstart, ystart = k
        caption = v['string']
        color = color_lookup[caption][0]
        plt.plot([], [], color=color, label=caption)

    plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
    plt.savefig('composite_captions.pdf', bbox_inches='tight')
    plt.savefig('composite_captions.png', bbox_inches='tight', facecolor=plt.gca().get_facecolor())

    with torch.no_grad():
        latents = stage_1(context, sampler, height=128, width=128, generator=generator, num_inference_steps=steps, guidance_scale=guidance_scale)

    with open(res_dir / save_file, "wb") as ff:
        pickle.dump(latents, ff)

    pickle.dump(latents, open("{}.p".format(save_file), "wb"))
    if mcmc_steps > 0 and isinstance(sampler, MCMCMHCorrSampler):
        sampler.save_stats_to_file(res_dir, save_file)

    plot_image(latents)
    plt.show()

    if stage2:
        stage_2 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16",
                                                    torch_dtype=torch.float16)
        stage_2.enable_xformers_memory_efficient_attention()
        stage_2.enable_model_cpu_offload()

        prompt = ""
        prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)
        latents = stage_2(image=latents, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds,
                          generator=generator, output_type="pt").images

        plot_image(latents)
        plt.show()
        pickle.dump(latents, open("{}_refined.p".format(save_file), "wb"))
