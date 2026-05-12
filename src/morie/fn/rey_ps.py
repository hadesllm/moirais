# morie.fn -- function file (hadesllm/morie)
"""Poisson regression. 'Waste no more time arguing what a good person should be. Be one. -- Marcus Aurelius'"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import gammaln

from ._containers import RegressionResult
from ._helpers import _validate_df


def poisson_regression(
    data: pd.DataFrame,
    *,
    y: str = "y",
    x: list[str] | str = "x",
) -> RegressionResult:
    """Poisson regression via maximum likelihood (scipy.optimize).

    :param data: DataFrame with count outcome and predictor columns.
    :param y: Name of the count outcome column.
    :param x: Predictor column name(s).
    :return: RegressionResult with coefficients on the log scale.
    """
    _validate_df(data, y)
    if isinstance(x, str):
        x = [x]
    _validate_df(data, *x)
    df = data[[y, *x]].dropna()
    Y = df[y].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(df)), df[x].to_numpy(dtype=float)])
    n, p = X.shape

    def neg_ll(beta):
        eta = X @ beta
        eta = np.clip(eta, -20, 20)
        mu = np.exp(eta)
        return -float(np.sum(Y * eta - mu - gammaln(Y + 1)))

    def grad(beta):
        eta = np.clip(X @ beta, -20, 20)
        mu = np.exp(eta)
        return -X.T @ (Y - mu)

    beta0 = np.zeros(p)
    res = optimize.minimize(neg_ll, beta0, jac=grad, method="L-BFGS-B")
    beta = res.x

    eta = np.clip(X @ beta, -20, 20)
    mu = np.exp(eta)
    W = np.diag(mu)
    try:
        cov = np.linalg.inv(X.T @ W @ X)
    except np.linalg.LinAlgError:
        cov = np.full((p, p), np.nan)
    se_arr = np.sqrt(np.maximum(np.diag(cov), 0))

    names = ["(Intercept)", *x]
    coefficients = {nm: float(b) for nm, b in zip(names, beta)}
    se_dict = {nm: float(s) for nm, s in zip(names, se_arr)}
    p_dict = {
        nm: float(2 * stats.norm.sf(abs(beta[i] / se_arr[i]))) if se_arr[i] > 0 else float("nan")
        for i, nm in enumerate(names)
    }

    return RegressionResult(
        method="Poisson",
        coefficients=coefficients,
        se=se_dict,
        p_values=p_dict,
        fitted=mu,
        n=n,
        k=p - 1,
    )


rey_ps = poisson_regression


def cheatsheet() -> str:
    return "poisson_regression({}) -> Poisson regression. 'Patience you must have, my young Padawa"
