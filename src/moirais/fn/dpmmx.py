# moirais.fn — function file (hadesllm/moirais)
"""Dirichlet process mixture model."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def dp_mixture_model(
    data: Union[list, np.ndarray],
    *,
    alpha: float = 1.0,
    K_max: int = 20,
    n_iter: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Dirichlet process mixture of Gaussians via truncated stick-breaking.

    :param data: Observed data (n,) or (n, d).
    :param alpha: Concentration parameter.
    :param K_max: Truncation level.
    :param n_iter: Number of Gibbs sweeps.
    :param seed: Random seed.
    :return: Dictionary with assignments, component_means, weights, n_active.

    References
    ----------
    Ishwaran, H. & James, L. F. (2001). *JASA*, 96(453), 161--173.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(data, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n, d = x.shape

    means = x[rng.choice(n, K_max, replace=True)].copy()
    variances = np.ones((K_max, d))
    V = rng.beta(1, alpha, size=K_max)
    V[-1] = 1.0
    weights = np.zeros(K_max)
    prod = 1.0
    for k in range(K_max):
        weights[k] = V[k] * prod
        prod *= (1 - V[k])

    assignments = np.zeros(n, dtype=int)

    for _ in range(n_iter):
        for i in range(n):
            log_probs = np.zeros(K_max)
            for k in range(K_max):
                diff = x[i] - means[k]
                log_probs[k] = np.log(weights[k] + 1e-30) - 0.5 * np.sum(diff ** 2 / (variances[k] + 1e-30))
            log_probs -= np.max(log_probs)
            probs = np.exp(log_probs)
            probs /= np.sum(probs)
            assignments[i] = rng.choice(K_max, p=probs)

        for k in range(K_max):
            members = x[assignments == k]
            nk = len(members)
            if nk > 0:
                means[k] = np.mean(members, axis=0)
                if nk > 1:
                    variances[k] = np.var(members, axis=0, ddof=1) + 1e-6
            else:
                means[k] = rng.standard_normal(d)
                variances[k] = np.ones(d)

        for k in range(K_max):
            nk = np.sum(assignments == k)
            n_gt = np.sum(assignments > k)
            V[k] = rng.beta(1 + nk, alpha + n_gt) if k < K_max - 1 else 1.0

        prod = 1.0
        for k in range(K_max):
            weights[k] = V[k] * prod
            prod *= (1 - V[k])

    n_active = len(set(assignments))

    return {
        "assignments": assignments.tolist(),
        "component_means": means[:n_active].tolist(),
        "weights": weights.tolist(),
        "n_active": n_active,
        "K_max": K_max,
        "n_iter": n_iter,
    }


dpmmx = dp_mixture_model


def cheatsheet() -> str:
    return "dp_mixture_model({}) -> Dirichlet process mixture model."
