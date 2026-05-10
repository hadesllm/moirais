# moirais.fn — function file (hadesllm/moirais)
"""Instrumental variables (2SLS). 'Real knowledge is to know the extent of one's ignorance. -- Confucius'"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._containers import RegressionResult
from ._helpers import _validate_df


def iv_2sls(
    data: pd.DataFrame,
    *,
    y: str = "outcome",
    d: str = "treatment",
    z: str = "instrument",
    x: list[str] | str | None = None,
) -> RegressionResult:
    """Two-stage least squares (2SLS) IV estimation."""
    _validate_df(data, y, d, z)
    cols = [y, d, z]
    if x is not None:
        if isinstance(x, str):
            x = [x]
        _validate_df(data, *x)
        cols.extend(x)
    else:
        x = []
    df = data[cols].dropna()
    Y = df[y].to_numpy(dtype=float)
    D = df[d].to_numpy(dtype=float)
    Z = df[z].to_numpy(dtype=float)
    n = len(df)

    X_exog = np.column_stack([np.ones(n)] + [df[c].to_numpy(dtype=float) for c in x]) if x else np.ones((n, 1))

    # Stage 1: D = gamma*Z + X*delta + v
    Z1 = np.column_stack([X_exog, Z])
    gamma_hat = np.linalg.lstsq(Z1, D, rcond=None)[0]
    D_hat = Z1 @ gamma_hat

    # Stage 2: Y = beta*D_hat + X*theta + u
    X2 = np.column_stack([D_hat, X_exog])
    beta_2sls = np.linalg.lstsq(X2, Y, rcond=None)[0]

    # Correct residuals use actual D, not D_hat
    X2_actual = np.column_stack([D, X_exog])
    resid = Y - X2_actual @ beta_2sls
    p = X2.shape[1]
    mse = float(np.sum(resid**2)) / (n - p)

    try:
        # 2SLS variance: sigma^2 * (X'PzX)^-1 where Pz = Z(Z'Z)^-1Z'
        PzX = np.column_stack([D_hat, X_exog])
        cov = mse * np.linalg.inv(PzX.T @ X2_actual)
    except np.linalg.LinAlgError:
        cov = np.full((p, p), np.nan)

    se_arr = np.sqrt(np.maximum(np.abs(np.diag(cov)), 0))
    names = [d] + (["(Intercept)"] + x if x else ["(Intercept)"])
    coefficients = {nm: float(b) for nm, b in zip(names, beta_2sls)}
    se_dict = {nm: float(s) for nm, s in zip(names, se_arr)}
    p_dict = {}
    for i, nm in enumerate(names):
        if se_arr[i] > 0:
            t_s = beta_2sls[i] / se_arr[i]
            p_dict[nm] = float(2 * stats.t.sf(abs(t_s), n - p))
        else:
            p_dict[nm] = float("nan")

    # First-stage F
    resid_1 = D - Z1 @ gamma_hat
    resid_1r = D - X_exog @ np.linalg.lstsq(X_exog, D, rcond=None)[0]
    ss_1 = np.sum(resid_1r**2) - np.sum(resid_1**2)
    f_stat = (ss_1 / 1) / (np.sum(resid_1**2) / (n - Z1.shape[1])) if np.sum(resid_1**2) > 0 else 0

    return RegressionResult(
        method="2SLS",
        coefficients=coefficients,
        se=se_dict,
        p_values=p_dict,
        n=n,
        k=p - 1,
        extra={"first_stage_F": float(f_stat)},
    )


iv = iv_2sls


def cheatsheet() -> str:
    return "iv_2sls({}) -> Instrumental variables (2SLS). 'Your eyes can deceive you. D"
