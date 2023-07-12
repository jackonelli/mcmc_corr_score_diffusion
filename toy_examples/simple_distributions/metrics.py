from statistics import NormalDist
import jax.numpy as jnp


def prob_gmm_independent_uniform(means, sigmas, bounds, weights=None):
    if weights is None:
        weights = jnp.ones(means.shape[0])
    total_prob = 0.
    for mean, std, w in zip(means, sigmas, weights):
        prob = list()
        for i, bound in enumerate(bounds):
            nor = NormalDist(mu=mean[i], sigma=std[i])
            prob.append(nor.cdf(bound[1]) - nor.cdf(bound[0]))
        total_prob += w * jnp.prod(jnp.array(prob))
    return total_prob


def compute_normalizing_constant(means, std, n_comp, pdf_outer, pdf_inner, bounds_outer, bounds_inner):
    prob_all = prob_gmm_independent_uniform(means, jnp.ones_like(means)*std, bounds_outer,
                                            jnp.ones(means.shape[0])/n_comp)
    prob_inner = prob_gmm_independent_uniform(means, jnp.ones_like(means)*std, bounds_inner,
                                              jnp.ones(means.shape[0])/n_comp)
    prob_outer = prob_all - prob_inner
    return pdf_outer * prob_outer + pdf_inner * prob_inner
