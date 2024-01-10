"""2D simulated composition dataset"""
import numpy as np
import torch as th


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

    def nll(self, x):
        in_x = np.abs(x[:, 0]) < self.x_bound
        in_y = np.abs(x[:, 1]) < self.y_bound
        in_ = th.logical_and(in_x, in_y)
        out = th.logical_not(in_)
        nll = th.empty((x.size(0),))
        # Samples in have uniform prob = 1 / area of support
        support_area = 4 * self.x_bound * self.y_bound
        nll[in_] = np.log(support_area)
        nll[out] = np.inf
        return nll.mean().item()


class GmmRadial:
    """Gaussian mixture model with Gaussians spread equiangularly on a ring with fixed radius"""

    def __init__(self, n_comp=8, std=0.03, radius=0.5):
        means_x = th.cos(2 * np.pi * th.linspace(0, (n_comp - 1) / n_comp, n_comp))
        means_y = th.sin(2 * np.pi * th.linspace(0, (n_comp - 1) / n_comp, n_comp))
        self.n_comp = n_comp
        self.means = radius * th.column_stack((means_x, means_y))
        self.std = std
        self.weights = th.ones(n_comp) / n_comp

    def sample(self, n_samples, shuffle=True):
        samples = []
        labels = []
        # Sample the number of element in each component.
        sample_group_sz = np.random.multinomial(n_samples, self.weights)
        assert sample_group_sz.sum() == n_samples

        for i in range(self.n_comp):
            # A component can be empty (have zero samples)
            if sample_group_sz[i] == 0:
                continue
            sample_group = self.means[i] + self.std * th.randn((sample_group_sz[i], 2))
            labels.append(i * th.ones((sample_group_sz[i],)))
            samples.append(sample_group)

        samples = th.concatenate(samples, dim=0)
        labels = th.concatenate(labels, dim=0)
        assert samples.shape == (n_samples, 2)
        assert labels.shape == (n_samples,)

        if shuffle:
            rand_order = np.random.permutation(n_samples)
            samples = samples[rand_order]
            labels = labels[rand_order]

        return samples, labels

    def nll(self, x: th.Tensor) -> float:
        """Compute NLL for GMM

        - log p(x) = - log sum_i w_i N(x; mu_i, std**2 I)
        """
        # Same std for all components
        log_normalisation = np.log(2 * self.std**2 * np.pi)
        # Some broadcasting trickery to get differences for all comp. means at once
        sq_diff = (x.unsqueeze(1) - self.means.unsqueeze(0)) ** 2
        # Sum over dimension d
        exp = -sq_diff.sum(dim=2) / (2 * self.std**2)
        # Create sum: log (w_i + exponent) => w_i * e^(exponent)
        weighted_exp = self.weights.log() + exp
        # Sum over mixture component i
        unnorm_log_pdf = th.logsumexp(weighted_exp, dim=1)
        # Mean of samples n
        return th.mean(-unnorm_log_pdf + log_normalisation).item()


import matplotlib.pyplot as plt


def test():
    samples, labels = GmmRadial(n_comp=1).sample(800)
    plt.scatter(samples[:, 0], samples[:, 1], c=labels)


if __name__ == "__main__":
    test()
