# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian nonparametric density estimation via Dirichlet process."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def dp_density(
    data: Union[list, np.ndarray],
    *,
    alpha: float = 1.0,
    n_iter: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Dirichlet process density estimation via the Polya urn (CRP) scheme.

    Each data point is assigned to a cluster; cluster parameters are
    drawn from a Normal-Inverse-Gamma base measure.

    :param data: Observed data (1-D array).
    :param alpha: Concentration parameter.
    :param n_iter: Number of Gibbs sweeps.
    :param seed: Random seed.
    :return: Dictionary with cluster assignments, means, n_clusters.

    References
    ----------
    Neal, R. M. (2000). *JCGS*, 9(2), 249--265.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(data, dtype=float).ravel()
    n = len(x)

    assignments = np.zeros(n, dtype=int)
    cluster_means = {0: float(np.mean(x))}
    cluster_vars = {0: float(np.var(x, ddof=1)) + 1e-6}

    for _ in range(n_iter):
        for i in range(n):
            old_k = assignments[i]
            counts = {}
            for j in range(n):
                if j != i:
                    k = assignments[j]
                    counts[k] = counts.get(k, 0) + 1

            log_probs = {}
            for k, nk in counts.items():
                mu_k = cluster_means.get(k, 0.0)
                var_k = cluster_vars.get(k, 1.0)
                ll = -0.5 * (x[i] - mu_k) ** 2 / var_k - 0.5 * np.log(var_k + 1e-30)
                log_probs[k] = np.log(nk) + ll

            new_k = max(assignments) + 1
            log_probs[new_k] = np.log(alpha) - 0.5 * x[i] ** 2

            keys = list(log_probs.keys())
            vals = np.array([log_probs[k] for k in keys])
            vals -= np.max(vals)
            probs = np.exp(vals)
            probs /= np.sum(probs)

            chosen = keys[rng.choice(len(keys), p=probs)]
            assignments[i] = chosen
            if chosen not in cluster_means:
                cluster_means[chosen] = x[i]
                cluster_vars[chosen] = 1.0

        for k in set(assignments):
            members = x[assignments == k]
            if len(members) > 0:
                cluster_means[k] = float(np.mean(members))
                cluster_vars[k] = float(np.var(members, ddof=1)) + 1e-6 if len(members) > 1 else 1.0

    unique_clusters = sorted(set(assignments))
    n_clusters = len(unique_clusters)

    return {
        "assignments": assignments.tolist(),
        "n_clusters": n_clusters,
        "cluster_means": {str(k): cluster_means[k] for k in unique_clusters},
        "alpha": alpha,
        "n_iter": n_iter,
    }


bnpar = dp_density


def cheatsheet() -> str:
    return "dp_density({}) -> Bayesian nonparametric density estimation via Dirichlet process."
