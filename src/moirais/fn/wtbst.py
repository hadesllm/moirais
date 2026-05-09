"""Weighted bootstrap (general)."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["wtbst"]


def wtbst(
    x: np.ndarray,
    statistic: callable | None = None,
    *,
    weight_func: callable | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> dict[str, Any]:
    r"""
    General weighted bootstrap.

    Instead of resampling, assigns random weights :math:`w_i` to
    observations and computes the weighted statistic. Allows arbitrary
    weight-generating distributions.

    :param x: 1-D array of observations.
    :param statistic: Function(x, weights) -> scalar. Default: weighted mean.
    :param weight_func: Function(rng, n) -> weights array. Default: Exp(1).
    :param n_boot: Bootstrap replications. Default 1000.
    :param alpha: Significance level. Default 0.05.
    :param seed: Random seed.
    :return: Dict with ``estimate``, ``se``, ``ci_lower``, ``ci_upper``,
        ``boot_distribution``, ``n``.
    :raises ValueError: If x is empty.

    References
    ----------
    Kosorok, M.R. (2008). Ch. 19. Springer.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("x must be non-empty.")

    n = x.size
    rng = np.random.default_rng(seed)

    if statistic is None:
        def statistic(data, w):
            return float(np.sum(w * data) / np.sum(w))

    if weight_func is None:
        def weight_func(rng_, n_):
            return rng_.exponential(1.0, size=n_)

    boot_dist = np.zeros(n_boot)
    for b in range(n_boot):
        w = weight_func(rng, n)
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
    return "wtbst({x}) -> General weighted bootstrap."
