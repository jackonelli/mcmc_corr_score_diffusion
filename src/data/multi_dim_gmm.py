import numpy as np
import torch as th
from torch.distributions.multivariate_normal import MultivariateNormal


class Gmm:
    """Gaussian mixture model in arbitrary dimensions"""

    def __init__(self, means, covs):
        self.num_comp, self.x_dim = means.size()
        self.means = means
        # NB: assumes diagonal covs
        self.covs = covs
        self.weights = th.ones(self.num_comp) / self.num_comp

    def sample(self, n_samples, shuffle=True):
        """TODO: be moved to a base class"""
        samples = []
        labels = []
        # Sample the number of element in each component.
        sample_group_sz = np.random.multinomial(n_samples, self.weights)
        assert sample_group_sz.sum() == n_samples

        for i in range(self.num_comp):
            # A component can be empty (have zero samples)
            if sample_group_sz[i] == 0:
                continue
            # Assuming diagonal covs, the Cholesky fact is simply a sqrt across the diagonal.
            # The transpose mult is simply to do it in batches over n_samples.
            sample_group = self.means[i] + th.randn((sample_group_sz[i], self.x_dim)) @ self.covs[i].sqrt()
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

    def full_nll(self, x: th.Tensor) -> float:
        """Compute NLL for GMM with general cov matrices

        - log p(x) = - log sum_i w_i N(x; mu_i, std**2 I)
        """
        ll_n = th.empty((self.num_comp, x.size(0)))
        for n in range(self.num_comp):
            distr = MultivariateNormal(loc=self.means[n], covariance_matrix=self.covs[n])
            ll_n[n, :] = distr.log_prob(x)
        # Transpose shenanigans to make the sum correct.
        # Note that this also means that the logsumexp is over dim=1
        weighted_exp = self.weights.log() + ll_n.T
        ll = th.logsumexp(weighted_exp, dim=1)
        return -ll.mean().item()

    def isotropic_nll(self, x: th.Tensor) -> float:
        """Compute NLL for GMM with isotropic cov matrix (Sigma = sigma^2 * I)
        where sigma_n = sigma for all components.

        # TODO: Generalise to diagonal cov. matrix

        - log p(x) = - log sum_i w_i N(x; mu_i, std**2 I)
        """
        # Same cov for all dimensions and components
        cov = self.covs[0][0][0].item()
        log_normalisation = self.x_dim / 2 * np.log(2 * np.pi) + self.x_dim * np.log(cov) / 2
        # Broadcasting trickery to get differences for all comp. means at once
        sq_diff = (x.unsqueeze(1) - self.means.unsqueeze(0)) ** 2
        # Sum over dimension d
        exp = -sq_diff.sum(dim=2) / (2 * cov)
        # Create sum: log (w_i) + log(exponent) => log(w_i * e^(exponent))
        weighted_exp = self.weights.log() + exp
        # Sum over mixture component i
        unnorm_log_pdf = th.logsumexp(weighted_exp, dim=1)
        # Mean of samples n
        return th.mean(-unnorm_log_pdf + log_normalisation).item()

    def conditional_nll(self, samples, classes):
        ll = 0.0
        for cl in th.arange(classes.max() + 1):
            mean = self.means[cl]
            cov = self.covs[cl]
            cond_samples = samples[classes == cl, :]
            pdf = MultivariateNormal(loc=mean, covariance_matrix=cov)
            ll += pdf.log_prob(cond_samples).sum().item()
        num_samples = classes.size(0)
        return -ll / num_samples


def generate_means(x_dim, num_comp):
    """Generate means on all unit vectors"""
    # Create list with all unit vectors
    pos_unit_vecs = th.eye(x_dim)
    # Add the negative versions.
    all_means = th.row_stack((pos_unit_vecs.clone(), -pos_unit_vecs))
    # Return a subset of all means, so that the number of means is num_comp
    assert num_comp <= 2 * x_dim, "Too many comp's for this assignment method (max = 2 * x_dim)"
    rand_order = np.random.permutation(2 * x_dim)
    return all_means[rand_order[:num_comp]]


def threshold_covs(x_dim: int, low_rank_dim: int, var_high: float, var_low: float = 1e-5):
    """Generate covariances which emulate a low rank manifold

    Creates a diagonal covariance matrix where a low_rank_dim number of elements have variance var_high,
    the remaining have essentially zero variance (a small positive number var_low)
    """
    vars = th.tensor(low_rank_dim * [var_high] + (x_dim - low_rank_dim) * [var_low])
    rand_order = np.random.permutation(x_dim)
    return th.diag(vars[rand_order])
