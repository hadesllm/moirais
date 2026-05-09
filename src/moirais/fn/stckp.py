"""Stick-breaking construction for Dirichlet process."""

from __future__ import annotations

from typing import Any

import numpy as np


def stick_breaking(
    alpha: float = 1.0,
    *,
    K: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Stick-breaking construction for a Dirichlet process.

    V_k ~ Beta(1, alpha), w_k = V_k * prod_{j<k} (1 - V_j).

    :param alpha: Concentration parameter.
    :param K: Truncation level.
    :param seed: Random seed.
    :return: Dictionary with weights, V (beta draws), cumulative_weight.

    References
    ----------
    Sethuraman, J. (1994). *Statistica Sinica*, 4(2), 639--650.
    """
    rng = np.random.default_rng(seed)
    V = rng.beta(1, alpha, size=K)
    V[-1] = 1.0

    weights = np.zeros(K)
    prod = 1.0
    for k in range(K):
        weights[k] = V[k] * prod
        prod *= (1 - V[k])

    return {
        "weights": weights.tolist(),
        "V": V.tolist(),
        "cumulative_weight": float(np.sum(weights)),
        "n_effective": int(np.sum(weights > 0.01)),
        "alpha": alpha,
        "K": K,
    }


stckp = stick_breaking


def cheatsheet() -> str:
    return "stick_breaking({}) -> Stick-breaking construction for Dirichlet process."
