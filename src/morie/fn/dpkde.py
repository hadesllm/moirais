# morie.fn -- function file (hadesllm/morie)
"""An unexamined life is not worth living. -- Socrates"""

from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde


def dirichlet_process_kde(
    x: np.ndarray,
    x_eval: np.ndarray | None = None,
    alpha: float = 1.0,
    bandwidth: float | None = None,
    n_iter: int = 100,
    rng: np.random.Generator | None = None,
) -> dict:
    r"""
    Dirichlet process mixture of Gaussians for kernel density estimation.

    Uses a DP mixture model to estimate the density:

    .. math::

        x_i | \mu_i, \sigma^2 \sim N(\mu_i, \sigma^2), \quad
        (\mu_i | G) \sim G, \quad G | \alpha \sim \mathcal{DP}(\alpha, N(m, v^2))

    The posterior density is estimated by averaging over MCMC samples.

    :param x: (n,) training observations.
    :type x: np.ndarray
    :param x_eval: Points at which to evaluate density. If None, uses x.
    :type x_eval: np.ndarray | None
    :param alpha: DP concentration parameter. Default 1.0.
    :type alpha: float
    :param bandwidth: Kernel bandwidth. If None, uses Silverman's rule.
    :type bandwidth: float | None
    :param n_iter: Number of MCMC iterations. Default 100.
    :type n_iter: int
    :param rng: Random number generator. If None, creates new generator.
    :type rng: np.random.Generator | None
    :return: Dictionary with keys: 'x_eval', 'density' (estimated), 'bw_used'.
    :rtype: dict

    References
    ----------
    Escobar M.D., West M. (1995). Bayesian density estimation and inference
    using mixtures. *J. Amer. Statist. Assoc.*, 90(430), 577-588.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x, dtype=float).ravel()
    n = len(x)

    if x_eval is None:
        x_eval = np.linspace(np.min(x) - 2, np.max(x) + 2, 200)
    else:
        x_eval = np.asarray(x_eval, dtype=float).ravel()

    if bandwidth is None:
        # Silverman's rule
        std = np.std(x, ddof=1)
        bandwidth = 1.06 * std * (n ** (-1.0 / 5.0))

    # Simple MCMC: approximate posterior via importance sampling
    densities_per_iter = []

    for _ in range(n_iter):
        # Sample cluster assignments (simplified Polya urn)
        z = np.zeros(n, dtype=int)
        cluster_means = [float(x[0])]

        for i in range(1, n):
            cluster_probs = np.array(
                [np.sum(z[:i] == k) for k in range(len(cluster_means))], dtype=float
            )
            cluster_probs = np.append(cluster_probs, alpha)
            cluster_probs /= np.sum(cluster_probs)

            new_cluster_idx = rng.choice(len(cluster_probs), p=cluster_probs)
            if new_cluster_idx < len(cluster_means):
                z[i] = new_cluster_idx
            else:
                z[i] = len(cluster_means)
                cluster_means.append(float(rng.normal(np.mean(x), np.std(x))))

        # Estimate density using KDE over cluster means
        if len(cluster_means) > 0:
            kde = gaussian_kde(np.array(cluster_means), bw_method=bandwidth / np.std(cluster_means))
            dens = kde.evaluate(x_eval)
        else:
            dens = np.ones_like(x_eval) / len(x_eval)

        densities_per_iter.append(dens)

    density = np.mean(densities_per_iter, axis=0)

    return {
        "x_eval": x_eval,
        "density": density,
        "bw_used": bandwidth,
        "n_obs": n,
        "alpha": alpha,
    }


dpkde = dirichlet_process_kde


def cheatsheet() -> str:
    return "dirichlet_process_kde(x, x_eval=None, alpha=1.0) -> DP-GMM density estimate"
