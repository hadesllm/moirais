# morie.fn — function file (hadesllm/morie)
"""EWMA volatility (RiskMetrics 1996)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["ewma_volatility"]


def ewma_volatility(x, lambda_=0.94):
    r"""RiskMetrics exponentially weighted moving-average variance.

    .. math::

        \sigma_t^2 = \lambda\,\sigma_{t-1}^2 + (1-\lambda)\,r_{t-1}^2.

    Parameters
    ----------
    x : array-like
        Return series.
    lambda_ : float, default 0.94
        RiskMetrics decay factor (0.94 daily / 0.97 monthly).

    Returns
    -------
    RichResult
        keys: ``conditional_variance``, ``conditional_volatility``,
        ``lambda``, ``n``, ``last_variance``, ``last_volatility``,
        ``method``.

    References
    ----------
    JP Morgan / Reuters (1996). RiskMetrics — Technical Document
    (4th ed.).
    """
    r = np.asarray(x, dtype=float).ravel()
    n = r.size
    if n < 2:
        raise ValueError(f"Need at least 2 observations, got {n}.")
    if not (0.0 < lambda_ < 1.0):
        raise ValueError(f"lambda must be in (0,1), got {lambda_}.")
    r2 = r ** 2
    s2 = np.zeros(n)
    s2[0] = r2[0]
    for t in range(1, n):
        s2[t] = lambda_ * s2[t - 1] + (1.0 - lambda_) * r2[t - 1]
    sig = np.sqrt(s2)
    return RichResult(payload={
        "conditional_variance": s2,
        "conditional_volatility": sig,
        "lambda": float(lambda_),
        "n": int(n),
        "last_variance": float(s2[-1]),
        "last_volatility": float(sig[-1]),
        "method": "EWMA RiskMetrics",
    })


def cheatsheet():
    return "ewtma: EWMA volatility (RiskMetrics 1996)."
