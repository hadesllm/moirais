"""Incremental validity: delta-R-squared when adding a new predictor."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp


def validity_incremental(y: np.ndarray, x_base: np.ndarray, x_new: np.ndarray, cdf=None) -> dict:
    """Incremental validity via hierarchical regression delta-R-squared.

    Compares R-squared of a base model (y ~ x_base) to the full model
    (y ~ x_base + x_new) and tests the change via partial F-test.

    Parameters
    ----------
    y : array-like
        Outcome variable (n,).
    x_base : array-like
        Base predictor(s) (n,) or (n, p1).
    x_new : array-like
        New predictor(s) to add (n,) or (n, p2).

    Returns
    -------
    dict
        Keys: ``r2_base``, ``r2_full``, ``delta_r2``, ``f_statistic``,
        ``p_value``, ``df1``, ``df2``, ``n``.

    References
    ----------
    Hunsley, J., & Meyer, G. J. (2003). The incremental validity of
    psychological testing and assessment. *Psychological Assessment*,
    15(4), 446--455.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    xb = np.asarray(x_base, dtype=np.float64)
    xn = np.asarray(x_new, dtype=np.float64)
    if xb.ndim == 1:
        xb = xb.reshape(-1, 1)
    if xn.ndim == 1:
        xn = xn.reshape(-1, 1)

    n = len(y)
    # Add intercept
    ones = np.ones((n, 1))
    X_base = np.hstack([ones, xb])
    X_full = np.hstack([ones, xb, xn])

    def _r2(X: np.ndarray) -> float:
        beta, res, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    r2_base = _r2(X_base)
    r2_full = _r2(X_full)
    delta_r2 = r2_full - r2_base

    p1 = X_base.shape[1]
    p2 = X_full.shape[1]
    df1 = p2 - p1
    df2 = n - p2

    if df1 > 0 and df2 > 0 and (1 - r2_full) > 0:
        f_stat = (delta_r2 / df1) / ((1 - r2_full) / df2)
        p_value = 1.0 - sp.f.cdf(f_stat, df1, df2)
    else:
        f_stat = np.nan
        p_value = np.nan

    return {
        "r2_base": float(r2_base),
        "r2_full": float(r2_full),
        "delta_r2": float(delta_r2),
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "df1": df1,
        "df2": df2,
        "n": n,
    }


def cheatsheet() -> str:
    return "validity_incremental({}) -> Incremental validity: delta-R-squared when adding a new pred"
