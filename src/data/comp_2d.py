"""2D simulated composition dataset"""
from typing import Tuple
import numpy as np
import torch as th

from src.data.multi_dim_gmm import Gmm


class Bar:
    """Uniform distribution in 2D"""

    def __init__(self, x_bound=0.2, y_bound=1.0):
        self.x_bound = x_bound
        self.y_bound = y_bound

    def sample(self, n_samples, _shuffle=True):
        data = np.random.uniform(-1, 1, (n_samples, 2))
        data[:, 0] = data[:, 0] * self.x_bound
        data[:, 1] = data[:, 1] * self.y_bound
        # Return x samples and dummy labels
        return th.tensor(data, dtype=th.float32), th.zeros((n_samples))

    def compute_support(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        in_x = th.abs(x[:, 0]) < self.x_bound
        in_y = th.abs(x[:, 1]) < self.y_bound
        in_ = th.logical_and(in_x, in_y)
        out = th.logical_not(in_)
        return in_, out

    def nll(self, x):
        in_, out = self.compute_support(x)
        nll = th.empty((x.size(0),))
        # Samples in have uniform prob = 1 / area of support
        support_area = 4 * self.x_bound * self.y_bound
        nll[in_] = np.log(support_area)
        nll[out] = np.inf
        return nll.mean().item()


class GmmRadial(Gmm):
    """Gaussian mixture model with Gaussians spread equiangularly on a ring with fixed radius"""

    def __init__(self, num_comp=8, std=0.03, radius=0.5):
        means_x = th.cos(2 * np.pi * th.linspace(0, (num_comp - 1) / num_comp, num_comp))
        means_y = th.sin(2 * np.pi * th.linspace(0, (num_comp - 1) / num_comp, num_comp))
        means = radius * th.column_stack((means_x, means_y))
        covs = [std**2 * th.eye(2) for _ in range(num_comp)]
        super().__init__(means, covs)
        self.std = std

    def nll(self, x: th.Tensor) -> float:
        """Compute NLL for GMM

        - log p(x) = - log sum_i w_i N(x; mu_i, std**2 I)
        """
        return self.isotropic_nll(x)


import matplotlib.pyplot as plt


def test():
    samples, labels = GmmRadial(num_comp=1).sample(800)
    plt.scatter(samples[:, 0], samples[:, 1], c=labels)


if __name__ == "__main__":
    test()
