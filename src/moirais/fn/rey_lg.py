# moirais.fn — function file (hadesllm/moirais)
"""The art of doing mathematics consists in finding that special case which contains all the germs of generality. — David Hilbert"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit

from ._containers import RegressionResult
from ._helpers import _validate_df


def logistic_regression(
    data: pd.DataFrame,
    *,
    y: str = "y",
    x: list[str] | str = "x",
    max_iter: int = 100,
    tol: float = 1e-8,
) -> RegressionResult:
    """Logistic regression via Iteratively Reweighted Least Squares (IRLS).

    :param data: DataFrame with binary outcome and predictor columns.
    :param y: Name of the binary outcome column (0/1).
    :param x: Predictor column name(s).
    :param max_iter: Maximum IRLS iterations.
    :param tol: Convergence tolerance.
    :return: RegressionResult with log-odds coefficients and pseudo-R-squared.
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
    p = k + 1

    beta = np.zeros(p)
    for _ in range(max_iter):
        eta = X_aug @ beta
        mu = expit(eta)
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        W = mu * (1 - mu)
        z = eta + (Y - mu) / W
        XtWX = X_aug.T @ (X_aug * W[:, None])
        XtWz = X_aug.T @ (W * z)
        try:
            beta_new = np.linalg.solve(XtWX, XtWz)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    mu = expit(X_aug @ beta)
    mu = np.clip(mu, 1e-10, 1 - 1e-10)
    W = mu * (1 - mu)

    try:
        cov = np.linalg.inv(X_aug.T @ (X_aug * W[:, None]))
    except np.linalg.LinAlgError:
        cov = np.full((p, p), np.nan)
    se_arr = np.sqrt(np.maximum(np.diag(cov), 0))

    names = ["(Intercept)", *x]
    coefficients = {nm: float(b) for nm, b in zip(names, beta)}
    se_dict = {nm: float(s) for nm, s in zip(names, se_arr)}
    p_dict = {}
    for i, nm in enumerate(names):
        if se_arr[i] > 0:
            z_stat = beta[i] / se_arr[i]
            p_dict[nm] = float(2 * stats.norm.sf(abs(z_stat)))
        else:
            p_dict[nm] = float("nan")

    ll = float(np.sum(Y * np.log(mu) + (1 - Y) * np.log(1 - mu)))
    y_bar = Y.mean()
    ll_null = float(n * (y_bar * np.log(y_bar + 1e-10) + (1 - y_bar) * np.log(1 - y_bar + 1e-10)))
    pseudo_r2 = 1 - ll / ll_null if ll_null != 0 else 0.0

    return RegressionResult(
        method="Logistic (IRLS)",
        coefficients=coefficients,
        se=se_dict,
        p_values=p_dict,
        r_squared=pseudo_r2,
        fitted=mu,
        n=n,
        k=k,
        extra={"log_likelihood": ll},
    )


rey_lg = logistic_regression


def cheatsheet() -> str:
    return "logistic_regression({}) -> Logistic regression via IRLS. 'Never tell me the odds.' -- H"
