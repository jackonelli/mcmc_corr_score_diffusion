from typing import Tuple
from statistics import NormalDist
import jax.numpy as jnp
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from src.simple_distributions.datasets import toy_gmm, bar


def prob_gmm_independent_uniform(means, sigmas, bounds, weights=None):
    if weights is None:
        weights = jnp.ones(means.shape[0])
    total_prob = 0.0
    for mean, std, w in zip(means, sigmas, weights):
        prob = list()
        for i, bound in enumerate(bounds):
            nor = NormalDist(mu=mean[i], sigma=std[i])
            prob.append(nor.cdf(bound[1]) - nor.cdf(bound[0]))
        total_prob += w * jnp.prod(jnp.array(prob))
    return total_prob


def compute_normalizing_constant(
    means, std, n_comp, pdf_outer, pdf_inner, bounds_outer, bounds_inner
):
    prob_all = prob_gmm_independent_uniform(
        means,
        jnp.ones_like(means) * std,
        bounds_outer,
        jnp.ones(means.shape[0]) / n_comp,
    )
    prob_inner = prob_gmm_independent_uniform(
        means,
        jnp.ones_like(means) * std,
        bounds_inner,
        jnp.ones(means.shape[0]) / n_comp,
    )
    prob_outer = prob_all - prob_inner
    return pdf_outer * prob_outer + pdf_inner * prob_inner


def fit_gmm(
    samples: np.ndarray, cov_type: str, n_comp=2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit Gaussian mixture model (GMM)

    Thin wrapper around scikit's GaussianMixture for our 2D product example.
    """
    allowed_cov_types = {"full", "tied", "diag", "spherical"}
    assert (
        cov_type in allowed_cov_types
    ), f"Incorrect covariance type, must be one of {allowed_cov_types}"
    model = GaussianMixture(n_components=n_comp, covariance_type=cov_type)
    gmm = model.fit(samples)
    return gmm.weights_, gmm.means_, gmm.covariances_


def gmm_params_metric(x_covs: np.ndarray, y_covs: np.ndarray) -> float:
    """GMM metric

    Compute a distance metric from two GMM's.
    The metric is the Frobenius norm of the difference in covariance,
    averaged over the mixture components.

    Args:
        x_covs: Covariance matrices (n_comp, *cov_dims) *Depends on the cov_type used in fit_gmm).
        y_covs: Same as x_covs but estimated from another sample.
    """
    n_comp = x_covs.shape[0]
    cov_metric = 0.0
    for x_cov_n, y_cov_n in zip(x_covs, y_covs):
        diff_ = x_cov_n - y_cov_n
        cov_metric += np.sqrt(np.sum(diff_ ** 2)).item()
    return cov_metric / n_comp


def gmm_metric(x_samples: np.ndarray, y_samples: np.ndarray, cov_type="diag") -> float:
    """GMM metric wrapper

    Estimates one GMM each for both X and Y samples,
    then computes the GMM based metric (see gmm_params_metric)
    """
    x_gmm, y_gmm = fit_gmm(x_samples, cov_type), fit_gmm(y_samples, cov_type)
    _, _, x_covs = x_gmm
    _, _, y_covs = y_gmm
    return gmm_params_metric(x_covs, y_covs)


def wasserstein_metric(
    x_samples: np.ndarray, y_samples: np.ndarray, p: float = 2.0
) -> float:
    """Wasserstein metric

    Computes the empirical Wasserstein p-distance between x_samples and y_samples
    by solving a linear assignment problem.

    Args:
        x_samples: samples
        y_samples: samples
        p: [0, inf) type of Wasserstein distance
    """

    d = cdist(x_samples, y_samples) ** p
    assignment = linear_sum_assignment(d)
    dist = (d[assignment].sum() / len(assignment)) ** (1.0 / p)
    return dist.item()


def ll_prod_metric(y_samples):
    n_comp = 8
    std = 0.03
    scale = 0.2
    r = 1.1
    prob_inside = 0.99

    # Load Data
    # Gaussian Mixture
    nll_gmm, _, means = toy_gmm(n_comp, std=std)
    bounds_outer = np.array([[-r, r], [-r, r]])
    bounds_inner = np.array([[-scale, scale], [-1.0, 1.0]])

    # Bar
    nll_bar, _, pdf_outer, pdf_inner = bar(scale=scale, r=r, prob_inside=prob_inside)
    c = compute_normalizing_constant(
        means, std, n_comp, pdf_outer, pdf_inner, bounds_outer, bounds_inner
    )
    return ll_prod(nll_gmm(y_samples), nll_bar(y_samples), c)


def ll_prod(nll_p1: np.ndarray, nll_p2: np.ndarray, c: float = 1.0) -> float:
    """Log-likelihood metric

    Evaluates LL for a product distribution with two components.

    Args:
        nll_p1: negative log-likelihood of distribution p1
        nll_p2: negative log-likelihood of distribution p2
        c: normalizing constant
    """

    ll = np.mean(-nll_p1 - nll_p2 - np.log(c))
    return ll.item()
