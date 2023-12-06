from typing import Optional
import random
import numpy as np
import torch as th


def set_seed(seed: Optional[int]):
    if seed is None:
        seed = th.initial_seed()
    else:
        print(f"Using manual seed '{seed}'")
    random.seed(seed)
    np.random.seed(seed)
    th.random.manual_seed(seed)
