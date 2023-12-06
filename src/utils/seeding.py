import torch as th


def set_seed(seed):
    if seed is None:
        seed = th.initial_seed()
    th.random.manual_seed(seed)
