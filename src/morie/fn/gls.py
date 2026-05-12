# morie.fn -- function file (hadesllm/morie)
"""Waste no more time arguing what a good person should be. Be one. -- Marcus Aurelius"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import RegressionResult
from ._helpers import _validate_df


def generalized_ls(
    data: pd.DataFrame,
    *,
    y: str = "y",
    x: list[str] | str = "x",
    max_iter: int = 50,
    tol: float = 1e-6,
) -> RegressionResult:
    """GLS via Cochrane-Orcutt for AR(1) errors. Dataset-agnostic.

    :param data: DataFrame with outcome and predictor columns.
    :param y: Name of the outcome column.
    :param x: Predictor column name(s).
    :param max_iter: Maximum Cochrane-Orcutt iterations.
    :param tol: Convergence tolerance.
    :return: RegressionResult with GLS coefficients and estimated rho.
    """
    _validate_df(data, y)
    if isinstance(x, str):
        x = [x]
    _validate_df(data, *x)
    df = data[[y, *x]].dropna()
    Y = df[y].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(df)), df[x].to_numpy(dtype=float)])
    n, p = X.shape

    # Initial OLS
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    rho = 0.0

    for _ in range(max_iter):
        resid = Y - X @ beta
        rho = float(np.corrcoef(resid[:-1], resid[1:])[0, 1]) if len(resid) > 2 else 0.0
        Y_t = Y[1:] - rho * Y[:-1]
        X_t = X[1:] - rho * X[:-1]
        beta_new = np.linalg.lstsq(X_t, Y_t, rcond=None)[0]
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    fitted = X @ beta
    resid = Y - fitted
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    names = ["(Intercept)", *x]
    coefficients = {nm: float(b) for nm, b in zip(names, beta)}

    return RegressionResult(
        method=f"GLS (rho={rho:.3f})",
        coefficients=coefficients,
        se={nm: float("nan") for nm in names},
        p_values={nm: float("nan") for nm in names},
        r_squared=r2,
        residuals=resid,
        fitted=fitted,
        n=n,
        k=p - 1,
        extra={"rho": rho},
    )


gls = generalized_ls


def cheatsheet() -> str:
    return "Waste no more time arguing what a good person should be. Be one. -- Marcus Aurelius"
