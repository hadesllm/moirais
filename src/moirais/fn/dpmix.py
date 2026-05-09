# moirais.fn — function file (hadesllm/moirais)
"""Patience is bitter, but its fruit is sweet. — Aristotle"""

from __future__ import annotations

import numpy as np


def dirichlet_process_mixture(
    x: np.ndarray,
    alpha: float = 1.0,
    n_iter: int = 1000,
    burn_in: int = 100,
    base_mean: float | None = None,
    base_cov: float = 1.0,
    rng: np.random.Generator | None = None,
) -> dict:
    r"""
    Fit a Dirichlet process mixture model using Gibbs sampling.

    Given observations :math:`x_1, \ldots, x_n`, the model assumes:

    .. math::

        x_i \sim N(\mu_{z_i}, \sigma^2), \quad \mu_k | G \sim G, \quad G | \alpha \sim \mathcal{DP}(\alpha, N(m_0, v_0^2))

    Gibbs sampler iterates:
    1. Update cluster assignments :math:`z_i` with Polya urn scheme
    2. Update cluster means :math:`\mu_k` from posterior

    :param x: (n,) observations.
    :type x: np.ndarray
    :param alpha: DP concentration parameter. Default 1.0.
    :type alpha: float
    :param n_iter: Number of MCMC iterations. Default 1000.
    :type n_iter: int
    :param burn_in: Burn-in samples. Default 100.
    :type burn_in: int
    :param base_mean: Base distribution mean. If None, uses empirical mean.
    :type base_mean: float | None
    :param base_cov: Base distribution variance. Default 1.0.
    :type base_cov: float
    :param rng: Random number generator. If None, creates new generator.
    :type rng: np.random.Generator | None
    :return: Dictionary with keys: 'cluster_assignments', 'cluster_means', 'n_clusters',
             'trace' (chain history), 'log_likelihood'.
    :rtype: dict

    References
    ----------
    Ghosal S., van der Vaart A.W. (2017). Fundamentals of Nonparametric Bayesian
    Inference. Cambridge University Press.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x, dtype=float).ravel()
    n = len(x)

    if base_mean is None:
        base_mean = float(np.mean(x))

    z = rng.integers(0, max(2, n // 5), size=n)  # Initialize cluster assignments
    cluster_means = {int(k): float(x[z == k].mean()) for k in np.unique(z)}

    trace = {"z": [], "means": [], "n_clusters": []}
    log_likelihoods = []

    for iteration in range(n_iter):
        # Update cluster assignments
        for i in range(n):
            # Remove i from its cluster
            z_i_old = z[i]
            if np.sum(z == z_i_old) == 1:
                del cluster_means[z_i_old]

            # Compute probabilities for existing clusters
            cluster_ids = list(cluster_means.keys())
            probs = []
            for k in cluster_ids:
                n_k = np.sum(z == k)
                likelihood = np.exp(-0.5 * (x[i] - cluster_means[k]) ** 2)
                prior = n_k / (alpha + n - 1)
                probs.append(prior * likelihood)

            # Probability of new cluster
            mu_new = rng.normal(base_mean, np.sqrt(base_cov))
            likelihood_new = np.exp(-0.5 * (x[i] - mu_new) ** 2)
            prior_new = alpha / (alpha + n - 1)
            probs.append(prior_new * likelihood_new)

            # Normalize and sample
            probs = np.array(probs, dtype=float)
            probs /= np.sum(probs)
            z_new = rng.choice(len(probs), p=probs)

            if z_new < len(cluster_ids):
                z[i] = cluster_ids[z_new]
            else:
                new_k = max(cluster_means.keys()) + 1 if cluster_means else 0
                z[i] = new_k
                cluster_means[new_k] = mu_new

        # Update cluster means
        for k in list(cluster_means.keys()):
            mask = z == k
            x_k = x[mask]
            n_k = np.sum(mask)

            if n_k > 0:
                post_mean = (n_k * np.mean(x_k) + base_mean / base_cov) / (n_k + 1.0 / base_cov)
                post_var = 1.0 / (n_k + 1.0 / base_cov)
                cluster_means[k] = float(rng.normal(post_mean, np.sqrt(post_var)))

        # Store trace
        if iteration >= burn_in:
            trace["z"].append(z.copy())
            trace["means"].append(dict(cluster_means))
            trace["n_clusters"].append(len(cluster_means))

            # Compute log-likelihood
            ll = 0.0
            for k, mean in cluster_means.items():
                mask = z == k
                ll += np.sum(-0.5 * (x[mask] - mean) ** 2)
            log_likelihoods.append(ll)

    final_z = np.array(trace["z"][-1]) if trace["z"] else z
    final_means = trace["means"][-1] if trace["means"] else cluster_means

    return {
        "cluster_assignments": final_z,
        "cluster_means": final_means,
        "n_clusters": len(final_means),
        "trace": trace,
        "log_likelihood": np.mean(log_likelihoods[-100:]) if log_likelihoods else np.nan,
        "alpha": alpha,
    }


dpmix = dirichlet_process_mixture


def cheatsheet() -> str:
    return "dirichlet_process_mixture(x, alpha=1.0, n_iter=1000) -> DP mixture via Gibbs"
