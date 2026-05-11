# morie.fn — function file (hadesllm/morie)
"""Hierarchical Dirichlet process for grouped data. 'Groups, understand them you must.'"""

from __future__ import annotations

import numpy as np


def hierarchical_dirichlet_process(
    x_groups: list[np.ndarray],
    alpha_0: float = 1.0,
    gamma: float = 1.0,
    n_iter: int = 100,
    rng: np.random.Generator | None = None,
) -> dict:
    r"""
    Hierarchical Dirichlet process for multi-group mixture modeling.

    Groups share a common DP, with each group having its own mixture:

    .. math::

        G_j | G_0 \sim \mathcal{DP}(\alpha_0, G_0), \quad
        G_0 | \gamma \sim \mathcal{DP}(\gamma, H)

    :param x_groups: List of (n_j,) observation arrays for each group.
    :param alpha_0: Group-level DP concentration.
    :param gamma: Global DP concentration.
    :param n_iter: Number of iterations.
    :param rng: Random number generator.
    :return: Dictionary with HDP results.
    """
    if rng is None:
        rng = np.random.default_rng()

    J = len(x_groups)
    group_assignments = [rng.integers(0, max(2, len(x_groups[j]) // 3), size=len(x_groups[j])) for j in range(J)]
    global_clusters = {}
    group_cluster_means = [{} for _ in range(J)]

    trace_n_global_clusters = []
    trace_group_n_clusters = [[] for _ in range(J)]

    for iteration in range(n_iter):
        # Update global cluster structure
        all_active_clusters = set()
        for j in range(J):
            all_active_clusters.update(set(group_assignments[j]))

        # Update cluster means for each group
        for j in range(J):
            for c in set(group_assignments[j]):
                x_j_c = x_groups[j][group_assignments[j] == c]
                group_cluster_means[j][c] = float(np.mean(x_j_c))

        # Update group-level cluster assignments (Gibbs)
        for j in range(J):
            for i in range(len(x_groups[j])):
                c_old = group_assignments[j][i]

                # Count clusters in group j
                cluster_counts = {c: np.sum(group_assignments[j] == c) for c in set(group_assignments[j])}
                if cluster_counts.get(c_old, 0) <= 1:
                    del cluster_counts[c_old]
                    if c_old in group_cluster_means[j]:
                        del group_cluster_means[j][c_old]

                # Likelihood for existing clusters
                likelihoods = []
                clusters = list(cluster_counts.keys())

                for c in clusters:
                    mu = group_cluster_means[j].get(c, 0)
                    lik = np.exp(-0.5 * (x_groups[j][i] - mu) ** 2)
                    prior = cluster_counts[c] / (alpha_0 + len(x_groups[j]) - 1)
                    likelihoods.append(prior * lik)

                # New cluster
                mu_new = rng.normal(0, 1)
                lik_new = np.exp(-0.5 * (x_groups[j][i] - mu_new) ** 2)
                prior_new = alpha_0 / (alpha_0 + len(x_groups[j]) - 1)
                likelihoods.append(prior_new * lik_new)

                likelihoods = np.array(likelihoods, dtype=float)
                likelihoods /= np.sum(likelihoods)

                c_new_idx = rng.choice(len(likelihoods), p=likelihoods)
                if c_new_idx < len(clusters):
                    group_assignments[j][i] = clusters[c_new_idx]
                else:
                    new_c = max(group_cluster_means[j].keys()) + 1 if group_cluster_means[j] else 0
                    group_assignments[j][i] = new_c
                    group_cluster_means[j][new_c] = mu_new

        # Track
        trace_n_global_clusters.append(len(all_active_clusters))
        for j in range(J):
            trace_group_n_clusters[j].append(len(set(group_assignments[j])))

    return {
        "trace_n_global_clusters": trace_n_global_clusters,
        "trace_group_n_clusters": trace_group_n_clusters,
        "final_n_global_clusters": len(all_active_clusters),
        "n_groups": J,
        "alpha_0": alpha_0,
        "gamma": gamma,
    }


hdprc = hierarchical_dirichlet_process


def cheatsheet() -> str:
    return "hierarchical_dirichlet_process(x_groups, alpha_0=1.0, gamma=1.0) -> HDP results"
