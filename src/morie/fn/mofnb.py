# morie.fn — function file (hadesllm/morie)
"""m-out-of-n bootstrap."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["mofnb"]


def mofnb(
    x: np.ndarray,
    statistic: callable | None = None,
    *,
    m: int | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> dict[str, Any]:
    r"""
    m-out-of-n bootstrap for irregular or non-Donsker problems.

    Resamples :math:`m < n` observations with replacement to ensure
    bootstrap consistency when the standard :math:`n`-out-of-:math:`n`
    bootstrap fails (e.g., for the maximum, cube-root asymptotics).

    Default :math:`m = \lfloor n^{2/3} \rfloor`.

    :param x: 1-D array of observations.
    :param statistic: Function(sample) -> scalar. Default: mean.
    :param m: Resample size. Default floor(n^(2/3)).
    :param n_boot: Bootstrap replications. Default 1000.
    :param alpha: Significance level. Default 0.05.
    :param seed: Random seed.
    :return: Dict with ``estimate``, ``se``, ``ci_lower``, ``ci_upper``,
        ``boot_distribution``, ``m``, ``n``.
    :raises ValueError: If x is empty or m > n.

    References
    ----------
    Kosorok, M.R. (2008). Ch. 20. Springer.
    Bickel, Gotze, van Zwet (1997). Resampling fewer than n observations.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("x must be non-empty.")

    n = x.size
    if m is None:
        m = max(2, int(n ** (2.0 / 3.0)))
    if m > n:
        raise ValueError(f"m must be <= n, got m={m}, n={n}.")

    rng = np.random.default_rng(seed)

    if statistic is None:
        def statistic(s):
            return float(np.mean(s))

    theta_hat = statistic(x)
    boot_dist = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=m, replace=True)
        boot_dist[b] = statistic(x[idx])

    scale = np.sqrt(m / n)
    scaled_dist = theta_hat + (boot_dist - theta_hat) / scale

    se = float(np.std(scaled_dist, ddof=1))
    ci_lower = float(np.quantile(scaled_dist, alpha / 2))
    ci_upper = float(np.quantile(scaled_dist, 1 - alpha / 2))

    return {
        "estimate": theta_hat,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "boot_distribution": boot_dist,
        "m": m,
        "n": n,
    }


def cheatsheet() -> str:
    return "mofnb({x}) -> m-out-of-n bootstrap."
