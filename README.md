# MCMC-Correction of Score-Based Diffusion Models for Model Composition

Implementation for the paper [MCMC-Correction of Score-Based Diffusion Models for Model Composition](https://arxiv.org/abs/2307.14012)

Large part of the code is based on the implementation for the paper [Reduce, reuse, recycle](https://arxiv.org/abs/2302.11552),
with original code [here](https://github.com/yilundu/reduce_reuse_recycle), commit `513361e60bb677dec75c086a234715f3db97ea51`

## Setup

```
# Get the source code
git clone git@github.com:FraunhoferChalmersCentre/mcmc_corr_score_diffusion.git
cd mcmc_corr_score_diffusion
# Install the environment.yml
```

## Experiments:

### 2D Composition

```
# Train all diffusion models and generate samples.
python r_3_comp_2d/train_script_product.py --exp_name=<SAVE_DIR> --num_retrains=5

# Compute metrics
python r_3_comp_2d/compute_metrics_product.py --samples_path=<SAVE_DIR>
```

### CIFAR-100

Train models
```
# Train score and energy diffusion models
python src/train_diff_cifar.py --ema --dataset=cifar100 --fixed_val
python src/train_diff_cifar.py --ema --dataset=cifar100 --energy --fixed_val
```

```
# Train classifiers (so-called classifier-full and classifier for evaluation)
python exp/train_class_t.py --dataset=cifar100 --ema
python exp/train_class.py --dataset=cifar100 --ema
```
Generate guided samples (Reverse, U-HMC, HMC, U-LA, and/or LA) for score and energy
```
# Note that the config files need to be adjusted - point at where the trained models are saved
python exp/sample_guided_diff.py --config exp/configs/cifar100_guided_energy_hmc.json
python exp/sample_guided_diff.py --config exp/configs/cifar100_guided_score_hmc.json
...
```
Compute metrics
```
# If computing FID is desired, first run, and decide where to save stats
python src/compute_fid --type_dataset1=cifar100_val --path_save_stats_1=<WHERE_TO_SAVE>

# Run script to compute metrics for generated samples
python src/compute_metrics.py --res_dir=<GENERATED_SAMPL_DIR> --path_fid=<cifar100_val.npz>
(<GENERATED_SAMPL_DIR> is the head folder where the subfolders are the results)
(<cifar100_val.npz> path to the file generated by compute_fid)
```

### ImageNet

Download models from https://github.com/openai/guided-diffusion
- 256x256_diffusion_uncond.pt
- 256x256_classifier.pt


Generate samples (Reverse or HMC)
```
# Add path to models in config
python src/sample_guided_diff.py --config exp/configs/imagenet_guided_score_hmc.json
```

Compute metrics
```
# If computing FID is desired, first download ILSVRC 2012 subset of ImageNet from https://image-net.org/download-images.php
python src/compute_fid --path_dataset1=<FOLDER_JPEG> --type_dataset1=jpeg --path_save_stats_1=<WHERE_TO_SAVE>

# Run script to compute metrics for generated samples
python src/compute_metrics.py --res_dir=<GENERATED_SAMPL_DIR> --path_fid=<imagenet_val.npz>
(<GENERATED_SAMPL_DIR> is the head folder where the subfolders are the results)
(<imagenet_val.npz> path to the file generated by compute_fid)
```

### Tapestry
In order to be able to use DeepFloyd IF-I-XL-v1.0, follow the instructions at https://huggingface.co/DeepFloyd/IF-I-XL-v1.0

Generate image
```
python src/image_tapestry.py --config exp/configs/tapestry.json
```

![alt text](Tapestry_stage2.png)