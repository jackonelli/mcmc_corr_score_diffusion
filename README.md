# MCMC-Correction for Diffusion Model Composition: Energy Approximation using Diffusion Models

Implementation for the paper [...]

The code is based on the implementation for the paper [Reduce, reuse, recycle](https://arxiv.org/abs/2302.11552),
with original code [here](https://github.com/yilundu/reduce_reuse_recycle), commit `513361e60bb677dec75c086a234715f3db97ea51`

## Setup

The python dependencies are managed by [`pipenv`](https://pipenv.pypa.io/en/latest/),
see `Pipfile` for the requirements.

```
# Get the source code
git clone git@github.com:jackonelli/post_lin_smooth.git
cd post_lin_smooth
# Start a shell with a python virtual env. with the needed deps.
pipenv shell
```

## Recreate experiments:

To reproduce the results in table 1, run

```
# Train models and generate samples.
python src/train_script_product.py --exp_name=<SAVE_DIR> --num_retrains=5
python src/compute_metrics_product.py.py --exp_name=<SAVE_DIR> --num_retrains=5
```

See script for more details.
