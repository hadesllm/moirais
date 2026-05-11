"""Semipartial (part) correlation."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import ESRes


def semipartial_corr(
    x: np.ndarray,
    y: np.ndarray,
    Z: np.ndarray,
) -> ESRes:
    """Semipartial correlation: x residualised on Z, correlated with raw y.

    Parameters
    ----------
    x, y : (n,) array-like
    Z : (n,) or (n, q) confounders

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
    beta = np.linalg.lstsq(Z_int, x, rcond=None)[0]
    rx = x - Z_int @ beta

    r_sp = float(np.corrcoef(rx, y)[0, 1])
    sd_rx = np.std(rx, ddof=1)
    sd_y = np.std(y, ddof=1)
    sr = r_sp * sd_rx / sd_y if sd_y > 0 else 0.0

    q = Z.shape[1]
    df = n - q - 2
    t_stat = sr * np.sqrt(df / (1 - sr**2 + 1e-12))
    p = float(2 * sp_stats.t.sf(abs(t_stat), df))

    return ESRes(
        measure="semipartial_r",
        estimate=float(sr),
        n=n,
        extra={"p_value": p, "t": float(t_stat), "df": df, "corr_resid_y": float(r_sp)},
    )


spcor = semipartial_corr


def cheatsheet() -> str:
    return "semipartial_corr({}) -> Semipartial (part) correlation."
