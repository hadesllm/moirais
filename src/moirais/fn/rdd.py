# moirais.fn — function file (hadesllm/moirais)
"""Regression discontinuity design. 'There has been an awakening.' -- Snoke"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._containers import RegressionResult
from ._helpers import _validate_df


def reg_discontinuity(
    data: pd.DataFrame,
    *,
    y: str = "outcome",
    r: str = "running",
    cutoff: float = 0.0,
    bandwidth: float | None = None,
) -> RegressionResult:
    """Sharp RDD: local linear regression around the cutoff."""
    _validate_df(data, y, r)
    df = data[[y, r]].dropna()
    R = df[r].to_numpy(dtype=float)
    Y = df[y].to_numpy(dtype=float)
    if bandwidth is None:
        bandwidth = 1.06 * R.std() * len(R) ** (-0.2)  # Silverman's rule
    mask = np.abs(R - cutoff) <= bandwidth
    if mask.sum() < 4:
        raise ValueError("Too few observations within bandwidth")

    R_local = R[mask] - cutoff
    Y_local = Y[mask]
    T = (R_local >= 0).astype(float)
    X = np.column_stack([np.ones(mask.sum()), R_local, T, R_local * T])
    beta = np.linalg.lstsq(X, Y_local, rcond=None)[0]
    resid = Y_local - X @ beta
    n = mask.sum()
    mse = float(np.sum(resid**2)) / (n - 4)
    try:
        cov = mse * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        cov = np.full((4, 4), np.nan)

    se_arr = np.sqrt(np.maximum(np.diag(cov), 0))
    names = ["intercept", "slope_left", "LATE", "slope_interaction"]
    coefficients = {nm: float(b) for nm, b in zip(names, beta)}
    se_dict = {nm: float(s) for nm, s in zip(names, se_arr)}
    p_dict = {}
    for i, nm in enumerate(names):
        if se_arr[i] > 0:
            t_stat = beta[i] / se_arr[i]
            p_dict[nm] = float(2 * stats.t.sf(abs(t_stat), n - 4))
        else:
            p_dict[nm] = float("nan")

    return RegressionResult(
        method="Sharp RDD",
        coefficients=coefficients,
        se=se_dict,
        p_values=p_dict,
        n=n,
        k=3,
        extra={"bandwidth": bandwidth, "cutoff": cutoff},
    )


rdd = reg_discontinuity


def cheatsheet() -> str:
    return "reg_discontinuity({}) -> Regression discontinuity design. 'There has been an awakenin"
