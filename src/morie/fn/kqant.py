# morie.fn -- function file (hadesllm/morie)
"""Kernel quantile estimator."""

from __future__ import annotations

import numpy as np
from ._richresult import RichResult

__all__ = ["kqant"]


def kqant(data: np.ndarray, probs: np.ndarray | None = None, cdf=None, *, bw: float | None = None, n_grid: int = 1024) -> dict:
    r"""
    Kernel quantile estimator.

    Inverts the kernel CDF estimator to obtain smoothed quantiles:

    .. math::

        \hat{Q}(p) = \hat{F}^{-1}(p) = \inf\{x : \hat{F}(x) \ge p\}

    Parameters
    ----------
    data : np.ndarray
        1-d array of observations.
    probs : array-like or None
        Probability levels in (0, 1). Default ``[0.25, 0.50, 0.75]``.
    bw : float or None
        Bandwidth. Silverman's rule if None.
    n_grid : int
        Internal CDF grid resolution.

    Returns
    -------
    dict
        ``probs``, ``quantiles``, ``bw``.

    Raises
    ------
    ValueError
        If probs outside (0, 1) or data is too small.

    References
    ----------
    Sheather, S. J. & Marron, J. S. (1990). Kernel quantile estimators.
        *JASA*, 85(410), 410-416.
    """
    from scipy.stats import norm

    data = np.asarray(data, dtype=float).ravel()
    n = data.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 observations.")

    if probs is None:
        probs = np.array([0.25, 0.50, 0.75])
    else:
        probs = np.asarray(probs, dtype=float).ravel()
    if np.any(probs <= 0) or np.any(probs >= 1):
        raise ValueError("All probabilities must be in (0, 1).")

    sigma = np.std(data, ddof=1)
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    s = min(sigma, iqr / 1.349) if iqr > 0 else sigma
    if bw is None:
        bw = max(0.9 * s * n ** (-0.2), 1e-10)
    if bw <= 0:
        raise ValueError(f"bw must be positive, got {bw}.")

    lo = data.min() - 4 * bw
    hi = data.max() + 4 * bw
    x_grid = np.linspace(lo, hi, n_grid)

    u = (x_grid[:, None] - data[None, :]) / bw
    cdf = norm.cdf(u).mean(axis=1)

    quantiles = np.interp(probs, cdf, x_grid)

    return RichResult(payload={"probs": probs, "quantiles": quantiles, "bw": bw})


def cheatsheet() -> str:
    return "kqant({data}) -> Kernel-smoothed quantile estimator."
