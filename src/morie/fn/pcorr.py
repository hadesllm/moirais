# morie.fn -- function file (hadesllm/morie)
"""Partial correlation."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import ESRes


def partial_correlation(
    x: np.ndarray,
    y: np.ndarray,
    Z: np.ndarray,
) -> ESRes:
    """Partial correlation of x and y controlling for Z.

    Parameters
    ----------
    x, y : (n,) array-like
    Z : (n,) or (n, q) array of confounders

    Returns
    -------
    ESRes
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    n = len(x)
    if n < Z.shape[1] + 3:
        raise ValueError("Need n >= q + 3.")

    Z_int = np.column_stack([np.ones(n), Z])

    def _resid(v):
        beta = np.linalg.lstsq(Z_int, v, rcond=None)[0]
        return v - Z_int @ beta

    rx = _resid(x)
    ry = _resid(y)
    r = float(np.corrcoef(rx, ry)[0, 1])

    q = Z.shape[1]
    df = n - q - 2
    t_stat = r * np.sqrt(df / (1 - r**2 + 1e-12))
    p = float(2 * sp_stats.t.sf(abs(t_stat), df))

    return ESRes(
        measure="partial_r",
        estimate=r,
        n=n,
        extra={"p_value": p, "t": float(t_stat), "df": df, "q": q},
    )


pcorr = partial_correlation


def cheatsheet() -> str:
    return "partial_correlation({}) -> Partial correlation."
