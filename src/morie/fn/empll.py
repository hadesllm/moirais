# morie.fn — function file (hadesllm/morie)
"""Empirical likelihood ratio."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import optimize, stats

__all__ = ["empll"]


def empll(x: np.ndarray, cdf=None, *, mu0: float = 0.0, alpha: float = 0.05) -> dict[str, Any]:
    r"""
    Compute the empirical likelihood ratio for the mean.

    The empirical likelihood is:

    .. math::

        \mathcal{L}(\mu) = \max \prod_{i=1}^n p_i
        \quad \text{s.t.} \quad \sum p_i = 1, \; \sum p_i X_i = \mu

    The log empirical likelihood ratio :math:`-2\log R(\mu_0)` is
    asymptotically :math:`\chi^2_1` under :math:`H_0: E[X] = \mu_0`.

    :param x: 1-D array of observations.
    :param mu0: Hypothesized mean. Default 0.
    :param alpha: Significance level. Default 0.05.
    :return: Dict with ``log_ratio``, ``p_value``, ``reject``, ``weights``,
        ``lagrange_multiplier``, ``n``.
    :raises ValueError: If x is empty.

    References
    ----------
    Kosorok, M.R. (2008). Ch. 14. Springer.
    Owen, A.B. (2001). *Empirical Likelihood*. Chapman & Hall/CRC.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("x must be non-empty.")

    n = x.size
    z = x - mu0

    if np.all(z == 0):
        return {
            "log_ratio": 0.0,
            "p_value": 1.0,
            "reject": False,
            "weights": np.full(n, 1.0 / n),
            "lagrange_multiplier": 0.0,
            "n": n,
        }

    # Empirical likelihood requires 0 to lie in the convex hull of z = x - mu0.
    # When all z have the same sign, H0 is rejected immediately with a Wald fallback.
    if np.min(z) > 0 or np.max(z) < 0:
        mean_x = float(np.mean(x))
        var_x = float(np.var(x, ddof=1))
        wald = n * (mean_x - mu0) ** 2 / max(var_x, 1e-12)
        p_value = float(1.0 - stats.chi2.cdf(wald, df=1))
        return {
            "log_ratio": float(wald),
            "p_value": p_value,
            "reject": p_value < alpha,
            "weights": np.full(n, 1.0 / n),
            "lagrange_multiplier": float("nan"),
            "n": n,
        }

    def dual(lam):
        vals = 1.0 / (1.0 + lam * z)
        if np.any(vals <= 0):
            return 1e10
        return -np.sum(np.log(vals))

    def dual_deriv(lam):
        vals = 1.0 + lam * z
        if np.any(vals <= 0):
            return 0.0
        return np.sum(z / vals)

    try:
        res = optimize.brentq(dual_deriv, -1.0 / (np.max(z) + 1e-10) + 1e-10, -1.0 / (np.min(z) - 1e-10) - 1e-10)
        lam = res
    except (ValueError, RuntimeError):
        lam = 0.0

    weights = 1.0 / (n * (1.0 + lam * z))
    weights = np.maximum(weights, 1e-300)
    weights /= np.sum(weights)

    log_ratio = -2.0 * np.sum(np.log(n * weights))
    p_value = float(1.0 - stats.chi2.cdf(max(log_ratio, 0), df=1))

    return {
        "log_ratio": float(log_ratio),
        "p_value": p_value,
        "reject": p_value < alpha,
        "weights": weights,
        "lagrange_multiplier": float(lam),
        "n": n,
    }


def cheatsheet() -> str:
    return "empll({x}) -> Empirical likelihood ratio for mean."
