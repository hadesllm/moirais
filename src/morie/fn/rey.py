# morie.fn — function file (hadesllm/morie)
"""OLS linear regression. 'The belonging you seek is not behind you, it is ahead.' -- Maz Kanata"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._containers import RegressionResult
from ._helpers import _validate_df


def linear_regression(
    data: pd.DataFrame,
    *,
    y: str = "y",
    x: list[str] | str = "x",
    alpha: float = 0.05,
) -> RegressionResult:
    """Ordinary least squares linear regression. Dataset-agnostic.

    :param data: DataFrame with outcome and predictor columns.
    :param y: Name of the outcome column.
    :param x: Predictor column name(s). String or list of strings.
    :param alpha: Significance level (unused, reserved for CI).
    :return: RegressionResult with coefficients, SE, p-values, R-squared.
    """
    _validate_df(data, y)
    if isinstance(x, str):
        x = [x]
    _validate_df(data, *x)
    df = data[[y, *x]].dropna()
    Y = df[y].to_numpy(dtype=float)
    X = df[x].to_numpy(dtype=float)
    n, k = X.shape
    X_aug = np.column_stack([np.ones(n), X])

    # OLS: beta = (X'X)^-1 X'Y
    try:
        beta, residuals_ss, rank, sv = np.linalg.lstsq(X_aug, Y, rcond=None)
    except np.linalg.LinAlgError:
        raise ValueError("Singular matrix -- check for perfect collinearity")

    fitted = X_aug @ beta
    resid = Y - fitted
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else r2

    # Standard errors (classical)
    df_resid = n - k - 1
    mse = ss_res / df_resid if df_resid > 0 else 0.0
    try:
        cov = mse * np.linalg.inv(X_aug.T @ X_aug)
    except np.linalg.LinAlgError:
        cov = np.full((k + 1, k + 1), np.nan)
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
        method="OLS",
        coefficients=coefficients,
        se=se_dict,
        p_values=p_dict,
        r_squared=r2,
        adj_r_squared=adj_r2,
        residuals=resid,
        fitted=fitted,
        n=n,
        k=k,
    )


rey = linear_regression


def cheatsheet() -> str:
    return "linear_regression({}) -> OLS linear regression. 'The belonging you seek is not behind"
