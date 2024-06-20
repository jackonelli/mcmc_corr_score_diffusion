"""
Code is based on https://github.com/yilundu/reduce_reuse_recycle commit 513361e60bb677dec75c086a234715f3db97ea51
"""
import matplotlib.pyplot as plt
from src.utils.tapestry_components import visualize_context, IFPipeline
from diffusers import DiffusionPipeline
from src.samplers.mcmc import AnnealedLAScoreSampler, AnnealedULAScoreSampler, MCMCMHCorrSampler
import torch
import pickle
import numpy as np
from src.utils.seeding import set_seed
from pathlib import Path
from datetime import datetime

# Set Seed
seed = 0

generator = torch.Generator('cuda').manual_seed(seed)
set_seed(seed)

has_cuda = torch.cuda.is_available()
device = torch.device('cpu' if not has_cuda else 'cuda')
print(device)

# initialize model
stage_1 = IFPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16,
                                     use_auth_token=True)
stage_1.enable_xformers_memory_efficient_attention()
stage_1.enable_model_cpu_offload()
stage_1.safety_checker = None


def plot_image(latents_):
    image = latents_[0].cpu().numpy().transpose(1, 2, 0)
    image = ((image + 1) / 2 * 255).astype(np.uint8)
    plt.imshow(image)


if __name__ == '__main__':
    # Tunable Parameters

    # Guidance Magnitude
    guidance_mag = 20.0

    """
    context = {
        (2, 0, 0):{'string':'An epic space battle', 'magnitude':guidance_mag},
        (1, 0, 0):{'string':'The starship Enterprise', 'magnitude':guidance_mag},
        (1, 64, 64):{'string':'A star destroyer from Star Wars', 'magnitude':guidance_mag},
    }
    
    context = {
        (1, 0, 0): {'string': 'The starship Enterprise', 'magnitude': guidance_mag},
        (1, 64, 0): {'string': 'An epic space battle', 'magnitude': guidance_mag},
        (1, 0, 64): {'string': 'An epic space battle', 'magnitude': guidance_mag},
        (1, 64, 64): {'string': 'A star destroyer from Star Wars', 'magnitude': guidance_mag},
        (1, 32, 0): {'string': 'An epic space battle', 'magnitude': guidance_mag},
        (1, 0, 32): {'string': 'An epic space battle', 'magnitude': guidance_mag},
        (1, 32, 32): {'string': 'An epic space battle', 'magnitude': guidance_mag},
        (1, 64, 32): {'string': 'An epic space battle', 'magnitude': guidance_mag},
        (1, 32, 64): {'string': 'An epic space battle', 'magnitude': guidance_mag},
    }
    
    context = {
        (1, 0, 0): {'string': 'A fearsome red dragon breathing fire', 'magnitude': guidance_mag},
        (1, 0, 64): {'string': 'A mysterious and powerful wizard casting a spell', 'magnitude': guidance_mag},
        (1, 64, 0): {'string': 'A majestic castle', 'magnitude': guidance_mag},
        (1, 64, 64): {'string': 'A brave and strong knight protecting himself with his shield', 'magnitude': guidance_mag},
        (1, 32, 0): {'string': 'Fantasy world', 'magnitude': guidance_mag},
        (1, 0, 32): {'string': 'An epic fantasy battle', 'magnitude': guidance_mag},
        (1, 32, 32): {'string': 'An epic fantasy battle', 'magnitude': guidance_mag},
        (1, 64, 32): {'string': 'An epic fantasy battle', 'magnitude': guidance_mag},
        (1, 32, 64): {'string': 'Fantasy world', 'magnitude': guidance_mag},
    }
    """
    context = {
        (1, 0, 0): {'string': 'The starship Enterprise shooting laser', 'magnitude': guidance_mag},
        (1, 64, 0): {'string': 'The moon made of cheese', 'magnitude': guidance_mag},
        (1, 0, 64): {'string': 'An epic space battle', 'magnitude': guidance_mag},
        (1, 64, 64): {'string': 'A star destroyer from Star Wars', 'magnitude': guidance_mag},
        (1, 32, 0): {'string': 'An epic space battle', 'magnitude': guidance_mag},
        (1, 0, 32): {'string': 'An epic space battle', 'magnitude': guidance_mag},
        (1, 32, 32): {'string': 'An epic space battle', 'magnitude': guidance_mag},
        (1, 64, 32): {'string': 'An epic space battle', 'magnitude': guidance_mag},
        (1, 32, 64): {'string': 'An epic space battle', 'magnitude': guidance_mag},
    }

    # Increase the number of MCMC steps run to sample between intermediate distributions
    mcmc_steps = 6

    # Steps sizes as a function of beta
    a = '1e0'
    step_sizes = stage_1.scheduler.betas * float(a)

    # Number of reverse steps
    steps = 1000

    # Save image
    time = datetime.now()
    time = time.strftime("%Y%m-%H%M")
    # save_file = 'space_seed0_1000_overlap_ULA_' + str(mcmc_steps) + '_' + a + '_' + time
    save_file = 'space_seed0_1000_overlap_LA_' + str(mcmc_steps) + '_' + a + '_' + time
    # save_file = 'fantasy_seed0_1000_reverse_' + time
    # save_file = 'space_seed0_1000_reverse_' + time

    # Stage 2
    stage2 = False

    # Construct Sampler
    sampler = AnnealedLAScoreSampler(mcmc_steps, step_sizes, None, 5)
    # sampler = AnnealedULAScoreSampler(mcmc_steps, step_sizes, None)

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
        latents = stage_1(context, sampler, height=128, width=128, generator=generator, num_inference_steps=steps, guidance_scale=guidance_mag)

    pickle.dump(latents, open("{}.p".format(save_file), "wb"))
    if mcmc_steps > 0 and isinstance(sampler, MCMCMHCorrSampler):
        sampler.save_stats_to_file(Path.cwd(), save_file + '.p')

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
