# morie.fn — function file (hadesllm/morie)
"""Glivenko-Cantelli test: sup|Fn - F| convergence."""

from __future__ import annotations

import numpy as np

__all__ = ["gcthm"]


def gcthm(x: np.ndarray, cdf=None, *, cdf_func: callable | None = None, alpha: float = 0.05) -> dict:
    r"""
    Test Glivenko-Cantelli convergence via sup|Fn(t) - F(t)|.

    Computes the Kolmogorov-Smirnov statistic :math:`D_n = \sup_t |\hat{F}_n(t) - F(t)|`
    and the DKW critical value at level ``alpha``.

    :param x: 1-D array of observations.
    :param cdf_func: True CDF. Default standard normal.
    :param alpha: Significance level. Default 0.05.
    :return: Dict with ``ks_stat``, ``critical_value``, ``converges`` (bool), ``n``.
    :raises ValueError: If x is empty or alpha not in (0,1).

    References
    ----------
    Kosorok, M.R. (2008). Ch. 5. Springer.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("x must be non-empty.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    if cdf_func is None:
        from scipy.stats import norm
        cdf_func = norm.cdf

    n = x.size
    x_sorted = np.sort(x)
    ecdf = np.arange(1, n + 1) / n
    ecdf_minus = np.arange(0, n) / n
    true_cdf = cdf_func(x_sorted)

    d_plus = np.max(ecdf - true_cdf)
    d_minus = np.max(true_cdf - ecdf_minus)
    ks_stat = max(d_plus, d_minus)

    critical_value = np.sqrt(np.log(2.0 / alpha) / (2.0 * n))

    return {
        "ks_stat": float(ks_stat),
        "critical_value": float(critical_value),
        "converges": bool(ks_stat <= critical_value),
        "n": n,
    }


def cheatsheet() -> str:
    return "gcthm({x}) -> Glivenko-Cantelli sup|Fn-F| convergence test."
