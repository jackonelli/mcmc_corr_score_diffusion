from typing import Optional
import random
import numpy as np
import torch as th


def set_seed(seed: Optional[int]):
    if seed is None:
        seed = int(th.randint(low=0, high=2**32 - 1, size=(1,)).item())
    else:
        print(f"Using manual seed '{seed}'")
    random.seed(seed)
    np.random.seed(seed)
    th.random.manual_seed(seed)


def test():
    set_seed(None)


if __name__ == "__main__":
    test()
