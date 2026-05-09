# moirais.fn — function file (hadesllm/moirais)
"""Lasso regression (L1). 'Your focus determines your reality.' -- Qui-Gon Jinn"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import RegressionResult
from ._helpers import _validate_df


def _soft_threshold(x, lam):
    """Soft-thresholding operator for L1 penalty."""
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)


def lasso_regression(
    data: pd.DataFrame,
    *,
    y: str = "y",
    x: list[str] | str = "x",
    lam: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> RegressionResult:
    """Lasso regression via coordinate descent. Dataset-agnostic.

    :param data: DataFrame with outcome and predictor columns.
    :param y: Name of the outcome column.
    :param x: Predictor column name(s).
    :param lam: L1 regularization parameter.
    :param max_iter: Maximum coordinate descent iterations.
    :param tol: Convergence tolerance.
    :return: RegressionResult with sparse coefficients.
    """
    _validate_df(data, y)
    if isinstance(x, str):
        x = [x]
    _validate_df(data, *x)
    df = data[[y, *x]].dropna()
    Y = df[y].to_numpy(dtype=float)
    X_raw = df[x].to_numpy(dtype=float)
    n, k = X_raw.shape

    x_mean = X_raw.mean(axis=0)
    x_std = X_raw.std(axis=0, ddof=0)
    x_std[x_std == 0] = 1.0
    X_s = (X_raw - x_mean) / x_std
    y_mean = Y.mean()
    Y_c = Y - y_mean

    beta = np.zeros(k)
    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(k):
            r_j = Y_c - X_s @ beta + X_s[:, j] * beta[j]
            z_j = X_s[:, j] @ r_j / n
            beta[j] = _soft_threshold(z_j, lam / n)
        if np.max(np.abs(beta - beta_old)) < tol:
            break

    beta_orig = beta / x_std
    intercept = y_mean - x_mean @ beta_orig
    fitted = X_raw @ beta_orig + intercept
    resid = Y - fitted
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    names = ["(Intercept)", *x]
    coefficients = {names[0]: float(intercept)}
    coefficients.update({nm: float(b) for nm, b in zip(x, beta_orig)})
    se_dict = {nm: float("nan") for nm in names}
    p_dict = {nm: float("nan") for nm in names}
    n_nonzero = int(np.sum(np.abs(beta_orig) > 1e-10))

    return RegressionResult(
        method=f"Lasso (lambda={lam})",
        coefficients=coefficients,
        se=se_dict,
        p_values=p_dict,
        r_squared=r2,
        residuals=resid,
        fitted=fitted,
        n=n,
        k=k,
        extra={"lambda": lam, "n_nonzero": n_nonzero},
    )


lasso = lasso_regression


def cheatsheet() -> str:
    return "_soft_threshold({}) -> Lasso regression (L1). 'Your focus determines your reality.'"
