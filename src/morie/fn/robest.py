# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Huber M-estimator of location via iteratively reweighted least squares."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult
from ._helpers import _extract_col


def robust_m_estimator(
    data: pd.DataFrame | np.ndarray,
    *,
    col: str = "x",
    c: float = 1.345,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> DescriptiveResult:
    """Huber M-estimator of location via iteratively reweighted least squares.

    Uses the Huber psi function with tuning constant *c* (default 1.345 gives
    95 percent asymptotic efficiency at the normal).  The algorithm iterates
    IRLS until convergence.

    Parameters
    ----------
    data : DataFrame or array
        Input data.
    col : str
        Column name if *data* is a DataFrame.
    c : float
        Huber tuning constant.  Smaller values give more robustness.
    tol : float
        Convergence tolerance on the location estimate.
    max_iter : int
        Maximum IRLS iterations.

    Returns
    -------
    DescriptiveResult
        With ``value`` = M-estimate of location.
    """
    x = _extract_col(data, col)
    if len(x) < 2:
        raise ValueError("Need at least 2 non-missing observations")
    mu = float(np.median(x))
    mad = float(np.median(np.abs(x - mu)))
    if mad == 0:
        return DescriptiveResult(
            name="Huber M-estimator",
            value=mu,
            extra={"iterations": 0, "converged": True, "c": c, "n": len(x), "mad": 0.0},
        )
    scale = mad / 0.6745
    for it in range(1, max_iter + 1):
        r = (x - mu) / scale
        w = np.where(np.abs(r) <= c, 1.0, c / np.abs(r))
        mu_new = float(np.sum(w * x) / np.sum(w))
        if abs(mu_new - mu) < tol:
            return DescriptiveResult(
                name="Huber M-estimator",
                value=mu_new,
                extra={"iterations": it, "converged": True, "c": c, "n": len(x), "scale": scale},
            )
        mu = mu_new
    return DescriptiveResult(
        name="Huber M-estimator",
        value=mu,
        extra={"iterations": max_iter, "converged": False, "c": c, "n": len(x), "scale": scale},
    )


def cheatsheet() -> str:
    return 'robust_m_estimator({}) -> Robust M-estimator.'
