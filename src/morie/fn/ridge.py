# morie.fn -- function file (hadesllm/morie)
"""Ridge regression (L2). 'Stay on target.' -- Gold Five"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import RegressionResult
from ._helpers import _validate_df


def ridge_regression(
    data: pd.DataFrame,
    *,
    y: str = "y",
    x: list[str] | str = "x",
    lam: float = 1.0,
    alpha: float = 0.05,
) -> RegressionResult:
    """Ridge regression with L2 penalty. Dataset-agnostic.

    :param data: DataFrame with outcome and predictor columns.
    :param y: Name of the outcome column.
    :param x: Predictor column name(s).
    :param lam: L2 regularization parameter (lambda).
    :param alpha: Significance level (reserved).
    :return: RegressionResult with penalized coefficients.
    """
    _validate_df(data, y)
    if isinstance(x, str):
        x = [x]
    _validate_df(data, *x)
    df = data[[y, *x]].dropna()
    Y = df[y].to_numpy(dtype=float)
    X_raw = df[x].to_numpy(dtype=float)
    n, k = X_raw.shape

    # Standardize X (not intercept)
    x_mean = X_raw.mean(axis=0)
    x_std = X_raw.std(axis=0, ddof=0)
    x_std[x_std == 0] = 1.0
    X_s = (X_raw - x_mean) / x_std
    y_mean = Y.mean()
    Y_c = Y - y_mean

    # Ridge: (X'X + lambda*I)^-1 X'y
    beta_s = np.linalg.solve(X_s.T @ X_s + lam * np.eye(k), X_s.T @ Y_c)

    # Unstandardize
    beta = beta_s / x_std
    intercept = y_mean - x_mean @ beta

    fitted = X_raw @ beta + intercept
    resid = Y - fitted
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Approximate effective df
    H = X_s @ np.linalg.inv(X_s.T @ X_s + lam * np.eye(k)) @ X_s.T
    df_eff = float(np.trace(H))
    mse = ss_res / (n - df_eff) if n > df_eff else ss_res / n

    names = ["(Intercept)", *x]
    coefficients = {names[0]: float(intercept)}
    coefficients.update({nm: float(b) for nm, b in zip(x, beta)})
    se_dict = {nm: float(np.sqrt(mse)) for nm in names}  # approximate
    p_dict = {nm: float("nan") for nm in names}  # no exact p-values for ridge

    return RegressionResult(
        method=f"Ridge (lambda={lam})",
        coefficients=coefficients,
        se=se_dict,
        p_values=p_dict,
        r_squared=r2,
        residuals=resid,
        fitted=fitted,
        n=n,
        k=k,
        extra={"lambda": lam, "df_effective": df_eff},
    )


ridge = ridge_regression


def cheatsheet() -> str:
    return "ridge_regression({}) -> Ridge regression (L2). 'Stay on target.' -- Gold Five"
