import numpy as np
import torch as th
from torch.utils.data import Dataset, DataLoader
from src.data.utils import collate_fn


class Gmm:
    """Gaussian mixture model in arbitrary dimensions"""

    def __init__(self, means, std):
        self.n_comp, self.x_dim = means.size()
        self.means = means
        # Assume same std for all comps
        self.std = std
        self.weights = th.ones(self.n_comp) / self.n_comp

    def sample(self, n_samples, shuffle=True):
        """TODO: be moved to a base class"""
        samples = []
        labels = []
        # Sample the number of element in each component.
        sample_group_sz = np.random.multinomial(n_samples, self.weights)
        assert sample_group_sz.sum() == n_samples

        for i in range(self.n_comp):
            # A component can be empty (have zero samples)
            if sample_group_sz[i] == 0:
                continue
            sample_group = self.means[i] + self.stds[i] * th.randn((sample_group_sz[i], self.x_dim))
            labels.append(i * th.ones((sample_group_sz[i],)))
            samples.append(sample_group)

        samples = th.concatenate(samples, dim=0)
        labels = th.concatenate(labels, dim=0)
        assert samples.shape == (n_samples, self.x_dim)
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
        log_normalisation = np.log(2 * self.stds**2 * np.pi)
        # Some broadcasting trickery to get differences for all comp. means at once
        sq_diff = (x.unsqueeze(1) - self.means.unsqueeze(0)) ** 2
        # Sum over dimension d
        exp = -sq_diff.sum(dim=2) / (2 * self.stds**2)
        # Create sum: log (w_i + exponent) => w_i * e^(exponent)
        weighted_exp = self.weights.log() + exp
        # Sum over mixture component i
        unnorm_log_pdf = th.logsumexp(weighted_exp, dim=1)
        # Mean of samples n
        return th.mean(-unnorm_log_pdf + log_normalisation).item()
