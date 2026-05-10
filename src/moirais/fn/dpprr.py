# moirais.fn — function file (hadesllm/moirais)
"""Dirichlet process prior predictive distribution. 'Knowledge itself is power. -- Bacon'"""
from __future__ import annotations

import numpy as np


def dirichlet_process_prior_predictive(
    alpha: float = 1.0,
    n_samples: int = 100,
    theta_dim: int = 1,
    base_dist: callable | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    r"""
    Sample from the Dirichlet process prior predictive distribution.

    Given a Dirichlet process :math:`\mathcal{DP}(\alpha, G_0)` with concentration
    :math:`\alpha` and base measure :math:`G_0`, samples are drawn as:

    .. math::

        \theta_i | G \sim G, \quad G | \alpha, G_0 \sim \mathcal{DP}(\alpha, G_0)

    This generates samples that exhibit clustering behavior due to the discreteness
    of the DP process.

    :param alpha: Concentration parameter (alpha > 0). Default 1.0.
    :type alpha: float
    :param n_samples: Number of samples to generate. Default 100.
    :type n_samples: int
    :param theta_dim: Dimension of samples from base distribution. Default 1.
    :type theta_dim: int
    :param base_dist: Callable that generates samples from base measure G_0.
                      If None, uses standard normal. Default None.
    :type base_dist: callable | None
    :param rng: Random number generator. If None, creates new generator.
    :type rng: np.random.Generator | None
    :return: Dictionary with keys: 'samples', 'cluster_assignments', 'unique_clusters'.
    :rtype: dict

    References
    ----------
    Antoniak C.E. (1974). Mixtures of Dirichlet processes with applications to
    Bayesian nonparametric problems. *Ann. Statist.*, 2(6), 1152-1174.
    """
    if rng is None:
        rng = np.random.default_rng()

    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")

    if base_dist is None:
        base_dist = lambda: rng.standard_normal(theta_dim)

    samples = []
    cluster_assignments = []
    unique_thetas = []

    for i in range(n_samples):
        if i == 0 or rng.uniform() < alpha / (alpha + i):
            # Draw new cluster from base distribution
            theta_new = np.asarray(base_dist(), dtype=float)
            if theta_new.ndim == 0:
                theta_new = theta_new.reshape(1)
            unique_thetas.append(theta_new)
            cluster_idx = len(unique_thetas) - 1
        else:
            # Reuse existing cluster proportional to frequency
            cluster_idx = rng.choice(len(unique_thetas))

        samples.append(unique_thetas[cluster_idx].copy())
        cluster_assignments.append(cluster_idx)

    samples = np.array(samples, dtype=float)
    cluster_assignments = np.array(cluster_assignments, dtype=int)
    unique_thetas = np.array(unique_thetas, dtype=float)

    return {
        "samples": samples,
        "cluster_assignments": cluster_assignments,
        "unique_clusters": unique_thetas,
        "n_clusters": len(unique_thetas),
        "alpha": alpha,
    }


dpprr = dirichlet_process_prior_predictive


def cheatsheet() -> str:
    return "dirichlet_process_prior_predictive(alpha=1.0, n_samples=100) -> DP prior predictive"
