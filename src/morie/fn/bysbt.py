# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian bootstrap."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["bysbt"]


def bysbt(
    x: np.ndarray,
    statistic: callable | None = None,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> dict[str, Any]:
    r"""
    Bayesian bootstrap (Rubin, 1981).

    Generates Dirichlet(1,...,1) weights and computes weighted statistics.
    The Bayesian bootstrap is the nonparametric posterior for the empirical
    measure under a Dirichlet process prior.

    :param x: 1-D array of observations.
    :param statistic: Function(x, weights) -> scalar. Default: weighted mean.
    :param n_boot: Bootstrap replications. Default 1000.
    :param alpha: Significance level. Default 0.05.
    :param seed: Random seed.
    :return: Dict with ``estimate``, ``se``, ``ci_lower``, ``ci_upper``,
        ``boot_distribution``, ``n``.
    :raises ValueError: If x is empty.

    References
    ----------
    Kosorok, M.R. (2008). Ch. 19. Springer.
    Rubin, D.B. (1981). The Bayesian bootstrap. *Annals of Statistics*.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("x must be non-empty.")

    n = x.size
    rng = np.random.default_rng(seed)

    if statistic is None:
        def statistic(data, w):
            return float(np.sum(w * data))

    boot_dist = np.zeros(n_boot)
    for b in range(n_boot):
        w = rng.dirichlet(np.ones(n))
        boot_dist[b] = statistic(x, w)

    estimate = float(np.mean(boot_dist))
    se = float(np.std(boot_dist, ddof=1))
    ci_lower = float(np.quantile(boot_dist, alpha / 2))
    ci_upper = float(np.quantile(boot_dist, 1 - alpha / 2))

    return {
        "estimate": estimate,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "boot_distribution": boot_dist,
        "n": n,
    }


def cheatsheet() -> str:
    return "bysbt({x}) -> Bayesian bootstrap."
