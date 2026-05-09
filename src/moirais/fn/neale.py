# moirais.fn — function file (hadesllm/moirais)
"""Neal Algorithm 8 for DP mixture MCMC sampling. 'Sample, Markov must.'"""

from __future__ import annotations

import numpy as np


def neal_algorithm_8(
    x: np.ndarray,
    alpha: float = 1.0,
    n_iter: int = 1000,
    aux_param: int = 1,
    rng: np.random.Generator | None = None,
) -> dict:
    r"""
    Neal's Algorithm 8 for DP mixture Gibbs sampling.

    Efficient algorithm using auxiliary parameters to avoid
    marginalizing out component parameters.

    .. math::

        x_i | z_i, \mu_{z_i} \sim N(\mu_{z_i}, 1)

    :param x: (n,) observations.
    :param alpha: DP concentration.
    :param n_iter: Number of iterations.
    :param aux_param: Number of auxiliary components.
    :param rng: Random number generator.
    :return: Dictionary with MCMC results.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x, dtype=float).ravel()
    n = len(x)

    # Initialize
    z = rng.integers(0, max(2, n // 5), size=n)
    cluster_means = {}

    trace_z = []
    trace_n_clusters = []
    trace_llik = []

    for iteration in range(n_iter):
        # Gibbs sampling of cluster assignments
        for i in range(n):
            # Count occupancy of each cluster
            unique_clusters = list(set(z))
            cluster_counts = {c: np.sum(z == c) for c in unique_clusters}

            # Remove current assignment
            z_old = z[i]
            if cluster_counts[z_old] == 1:
                del cluster_counts[z_old]
                if z_old in cluster_means:
                    del cluster_means[z_old]

            # Compute likelihood for each cluster
            likelihoods = []
            for c in cluster_counts:
                mu = cluster_means.get(c, 0)
                likelihood = np.exp(-0.5 * (x[i] - mu) ** 2)
                prior = cluster_counts[c] / (alpha + n - 1)
                likelihoods.append(prior * likelihood)

            # Likelihood for new cluster
            mu_new = rng.normal(0, 1)
            likelihood_new = np.exp(-0.5 * (x[i] - mu_new) ** 2)
            prior_new = alpha / (alpha + n - 1)
            likelihoods.append(prior_new * likelihood_new)

            # Sample new cluster assignment
            likelihoods = np.array(likelihoods, dtype=float)
            likelihoods /= np.sum(likelihoods)

            new_cluster_idx = rng.choice(len(likelihoods), p=likelihoods)
            clusters = list(cluster_counts.keys()) + [max(cluster_means.keys()) + 1 if cluster_means else 0]

            z[i] = clusters[new_cluster_idx]

        # Update cluster means
        for c in set(z):
            x_c = x[z == c]
            cluster_means[c] = float(np.mean(x_c))

        # Record trace
        trace_z.append(z.copy())
        trace_n_clusters.append(len(set(z)))

        # Log-likelihood
        llik = 0.0
        for c in set(z):
            mu = cluster_means[c]
            llik += np.sum(-0.5 * (x[z == c] - mu) ** 2)
        trace_llik.append(llik)

    return {
        "trace_z": trace_z,
        "trace_n_clusters": trace_n_clusters,
        "trace_llik": trace_llik,
        "final_assignment": z,
        "final_n_clusters": len(set(z)),
        "final_means": cluster_means,
        "alpha": alpha,
    }


neale = neal_algorithm_8


def cheatsheet() -> str:
    return "neal_algorithm_8(x, alpha=1.0, n_iter=1000) -> DP mixture MCMC results"
