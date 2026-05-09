# moirais.fn — function file (hadesllm/moirais)
"""DP mixture density estimation: posterior mean. 'Wise, you are becoming.'"""

from __future__ import annotations

import numpy as np


def dirichlet_process_mixture_density(
    x: np.ndarray,
    x_eval: np.ndarray | None = None,
    alpha: float = 1.0,
    n_iter: int = 100,
    rng: np.random.Generator | None = None,
) -> dict:
    r"""
    Estimate density via DP mixture using posterior mean.

    Runs Gibbs sampler for DP-GMM, then averages component estimates.

    .. math::

        \hat{f}(x) = \frac{1}{M} \sum_{m=1}^{M} \sum_{k=1}^{K_m} \pi_k^{(m)} \mathcal{N}(x | \mu_k^{(m)}, 1)

    :param x: (n,) observations.
    :param x_eval: Evaluation points. If None, uses data quantiles.
    :param alpha: DP concentration.
    :param n_iter: MCMC iterations.
    :param rng: Random number generator.
    :return: Dictionary with 'x_eval', 'density', 'n_components_trace'.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x, dtype=float).ravel()
    n = len(x)

    if x_eval is None:
        x_eval = np.quantile(x, np.linspace(0.01, 0.99, 100))
    else:
        x_eval = np.asarray(x_eval, dtype=float).ravel()

    # Initialize cluster assignments
    z = rng.integers(0, max(2, n // 5), size=n)
    cluster_means = {}
    cluster_weights = {}
    densities = []
    n_components_trace = []

    for iteration in range(n_iter):
        # Update cluster assignments (Polya urn)
        for i in range(n):
            cluster_ids = list(set(z[:i]) if i > 0 else [])
            probs = np.array([np.sum(z[:i] == k) for k in cluster_ids], dtype=float)
            probs = np.append(probs, alpha)
            probs /= np.sum(probs)

            new_cluster = rng.choice(len(probs), p=probs)
            if new_cluster < len(cluster_ids):
                z[i] = cluster_ids[new_cluster]
            else:
                z[i] = max(z[:i]) + 1 if i > 0 else 0

        # Update cluster means
        for k in set(z):
            mask = z == k
            x_k = x[mask]
            if len(x_k) > 0:
                cluster_means[k] = float(np.mean(x_k))
                cluster_weights[k] = np.sum(mask) / n

        # Compute density at x_eval
        dens = np.zeros_like(x_eval)
        for k, mu in cluster_means.items():
            w = cluster_weights.get(k, 0)
            dens += w * np.exp(-0.5 * (x_eval - mu) ** 2)

        densities.append(dens / np.sqrt(2 * np.pi))
        n_components_trace.append(len(cluster_means))

    # Average over last half of iterations (post-burn-in)
    burn_in = max(1, n_iter // 2)
    density = np.mean(densities[burn_in:], axis=0)

    return {
        "x_eval": x_eval,
        "density": density,
        "alpha": alpha,
        "n_components_trace": n_components_trace,
        "final_n_components": len(cluster_means),
    }


dpmdn = dirichlet_process_mixture_density


def cheatsheet() -> str:
    return "dirichlet_process_mixture_density(x, x_eval=None, alpha=1.0) -> DP-GMM density"
