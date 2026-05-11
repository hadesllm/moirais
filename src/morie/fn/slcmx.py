"""Slice sampler for DP mixtures. 'Slice the probability, you must.'"""

from __future__ import annotations

import numpy as np


def slice_sampling_dp_mixture(
    x: np.ndarray,
    alpha: float = 1.0,
    n_iter: int = 1000,
    rng: np.random.Generator | None = None,
) -> dict:
    r"""
    Slice sampling for DP mixture models.

    Uses auxiliary slice variable to avoid explicitly iterating
    over all potential clusters.

    .. math::

        x_i | z_i, \mu_{z_i} \sim N(\mu_{z_i}, 1), \quad
        u_i | z_i \sim \text{Uniform}(0, w_{z_i})

    :param x: (n,) observations.
    :param alpha: DP concentration.
    :param n_iter: Number of iterations.
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
    cluster_weights = {}

    trace_z = []
    trace_n_clusters = []

    for iteration in range(n_iter):
        # Slice sampling of cluster assignments
        for i in range(n):
            z_old = z[i]

            # Remove current assignment
            cluster_counts = {c: np.sum(z == c) for c in set(z)}
            if cluster_counts.get(z_old, 0) <= 1:
                if z_old in cluster_means:
                    del cluster_means[z_old]
                if z_old in cluster_weights:
                    del cluster_weights[z_old]

            # Slice variable: sample uniform threshold
            n_i = len(set(z))
            u_i = rng.uniform(0, alpha / (alpha + n_i))

            # Find clusters with weight > u_i
            active_clusters = []
            for c in set(z):
                if c in cluster_weights and cluster_weights[c] > u_i:
                    active_clusters.append(c)

            # If no active clusters, create new one
            if len(active_clusters) == 0:
                new_c = max(cluster_means.keys()) + 1 if cluster_means else 0
                cluster_means[new_c] = float(rng.normal(0, 1))
                cluster_weights[new_c] = alpha / (alpha + n)
                active_clusters = [new_c]

            # Sample from active clusters
            likelihoods = []
            for c in active_clusters:
                mu = cluster_means.get(c, 0)
                likelihood = np.exp(-0.5 * (x[i] - mu) ** 2)
                likelihoods.append(likelihood)

            likelihoods = np.array(likelihoods, dtype=float)
            likelihoods /= np.sum(likelihoods)

            z[i] = active_clusters[rng.choice(len(active_clusters), p=likelihoods)]

        # Update cluster weights (approximate)
        for c in set(z):
            cluster_weights[c] = np.sum(z == c) / n

        # Update cluster means
        for c in set(z):
            x_c = x[z == c]
            cluster_means[c] = float(np.mean(x_c))

        trace_z.append(z.copy())
        trace_n_clusters.append(len(set(z)))

    return {
        "trace_z": trace_z,
        "trace_n_clusters": trace_n_clusters,
        "final_assignment": z,
        "final_n_clusters": len(set(z)),
        "alpha": alpha,
    }


slcmx = slice_sampling_dp_mixture


def cheatsheet() -> str:
    return "slice_sampling_dp_mixture(x, alpha=1.0, n_iter=1000) -> slice sampler results"
