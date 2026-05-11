# morie.fn — function file (hadesllm/morie)
"""Regression discontinuity at age cutoff."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import ESRes


def otis_rdd_age(
    df,
    *,
    outcome_col: str = "outcome",
    running_var: str = "age",
    cutoff: float = 18.0,
    bandwidth: float | None = None,
) -> ESRes:
    """Sharp RDD at an age cutoff.

    Parameters
    ----------
    df : DataFrame
    outcome_col : str
    running_var : str
    cutoff : float
    bandwidth : float, optional

    Returns
    -------
    ESRes
    """
    y = np.asarray(df[outcome_col], dtype=float)
    x = np.asarray(df[running_var], dtype=float)
    if bandwidth is not None:
        mask = np.abs(x - cutoff) <= bandwidth
        y, x = y[mask], x[mask]
    n = len(y)
    if n < 4:
        return ESRes(measure="otis_rdd_age", estimate=0.0, n=n)
    xc = x - cutoff
    treat = (xc >= 0).astype(float)
    X = np.column_stack([np.ones(n), xc, treat, xc * treat])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    tau = float(beta[2])
    resid = y - X @ beta
    sigma2 = np.sum(resid**2) / max(n - 4, 1)
    se = np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[2, 2])
    return ESRes(
        measure="otis_rdd_age",
        estimate=tau,
        ci_lower=tau - 1.96 * float(se),
        ci_upper=tau + 1.96 * float(se),
        se=float(se),
        n=n,
    )


ordd1 = otis_rdd_age


def cheatsheet() -> str:
    return "otis_rdd_age({}) -> Regression discontinuity at age cutoff."
