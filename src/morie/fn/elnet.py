# morie.fn — function file (hadesllm/morie)
"""Elastic net regression. 'Everything flows. -- Heraclitus'"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import RegressionResult
from ._helpers import _validate_df


def _soft_threshold(x, lam):
    """Soft-thresholding operator."""
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)


def elastic_net(
    data: pd.DataFrame,
    *,
    y: str = "y",
    x: list[str] | str = "x",
    lam: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> RegressionResult:
    """Elastic net: alpha*L1 + (1-alpha)*L2 via coordinate descent. Dataset-agnostic.

    :param data: DataFrame with outcome and predictor columns.
    :param y: Name of the outcome column.
    :param x: Predictor column name(s).
    :param lam: Overall regularization strength.
    :param l1_ratio: Mixing parameter (1=Lasso, 0=Ridge).
    :param max_iter: Maximum iterations.
    :param tol: Convergence tolerance.
    :return: RegressionResult with coefficients.
    """
    _validate_df(data, y)
    if isinstance(x, str):
        x = [x]
    _validate_df(data, *x)
    df = data[[y, *x]].dropna()
    Y = df[y].to_numpy(dtype=float)
    X_raw = df[x].to_numpy(dtype=float)
    n, k = X_raw.shape

    x_mean, x_std = X_raw.mean(0), X_raw.std(0, ddof=0)
    x_std[x_std == 0] = 1.0
    X_s = (X_raw - x_mean) / x_std
    y_mean = Y.mean()
    Y_c = Y - y_mean

    beta = np.zeros(k)
    l1 = lam * l1_ratio
    l2 = lam * (1 - l1_ratio)

    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(k):
            r_j = Y_c - X_s @ beta + X_s[:, j] * beta[j]
            z_j = X_s[:, j] @ r_j / n
            beta[j] = _soft_threshold(z_j, l1 / n) / (1 + l2 / n)
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

    return RegressionResult(
        method=f"ElasticNet (lam={lam}, l1_ratio={l1_ratio})",
        coefficients=coefficients,
        se={nm: float("nan") for nm in names},
        p_values={nm: float("nan") for nm in names},
        r_squared=r2,
        residuals=resid,
        fitted=fitted,
        n=n,
        k=k,
    )


elnet = elastic_net


def cheatsheet() -> str:
    return "_soft_threshold({}) -> Elastic net regression. 'In my experience, there is no such "
