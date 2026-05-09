"""Weighted least squares. 'Size matters not.'"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._containers import RegressionResult
from ._helpers import _validate_df


def weighted_ls(
    data: pd.DataFrame,
    *,
    y: str = "y",
    x: list[str] | str = "x",
    w: str = "weight",
) -> RegressionResult:
    """Weighted least squares regression. Dataset-agnostic.

    :param data: DataFrame with outcome, predictor, and weight columns.
    :param y: Name of the outcome column.
    :param x: Predictor column name(s).
    :param w: Name of the weight column.
    :return: RegressionResult with WLS coefficients and standard errors.
    """
    _validate_df(data, y, w)
    if isinstance(x, str):
        x = [x]
    _validate_df(data, *x)
    df = data[[y, w, *x]].dropna()
    Y = df[y].to_numpy(dtype=float)
    W = df[w].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(df)), df[x].to_numpy(dtype=float)])
    n, p = X.shape

    W_sqrt = np.sqrt(np.maximum(W, 0))
    X_w = X * W_sqrt[:, None]
    Y_w = Y * W_sqrt
    beta = np.linalg.lstsq(X_w, Y_w, rcond=None)[0]

    fitted = X @ beta
    resid = Y - fitted
    ss_res = float(np.sum(W * resid**2))
    ss_tot = float(np.sum(W * (Y - np.average(Y, weights=W)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    df_resid = n - p
    mse = ss_res / df_resid if df_resid > 0 else 0.0
    try:
        cov = mse * np.linalg.inv(X_w.T @ X_w)
    except np.linalg.LinAlgError:
        cov = np.full((p, p), np.nan)
    se_arr = np.sqrt(np.maximum(np.diag(cov), 0))

    names = ["(Intercept)", *x]
    coefficients = {nm: float(b) for nm, b in zip(names, beta)}
    se_dict = {nm: float(s) for nm, s in zip(names, se_arr)}
    p_dict = {}
    for i, nm in enumerate(names):
        if se_arr[i] > 0 and df_resid > 0:
            t_stat = beta[i] / se_arr[i]
            p_dict[nm] = float(2 * stats.t.sf(abs(t_stat), df_resid))
        else:
            p_dict[nm] = float("nan")

    return RegressionResult(
        method="WLS",
        coefficients=coefficients,
        se=se_dict,
        p_values=p_dict,
        r_squared=r2,
        residuals=resid,
        fitted=fitted,
        n=n,
        k=p - 1,
    )


wls = weighted_ls


def cheatsheet() -> str:
    return "weighted_ls({}) -> Weighted least squares. 'Size matters not.'"
