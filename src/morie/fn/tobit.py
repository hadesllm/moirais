"""Tobit censored regression. 'Your eyes can deceive you. --'"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

from ._containers import RegressionResult
from ._helpers import _validate_df


def tobit_model(
    data: pd.DataFrame,
    *,
    y: str = "y",
    x: list[str] | str = "x",
    lower: float = 0.0,
) -> RegressionResult:
    """Tobit (Type I) censored regression via MLE.

    Estimates a linear model where the dependent variable is censored
    at *lower*.  Uses the Tobin (1958) log-likelihood with Nelder-Mead
    optimisation.

    :param data: DataFrame with outcome and predictor columns.
    :param y: Name of the outcome column.
    :param x: Predictor column name(s).
    :param lower: Left-censoring threshold (default 0).
    :return: RegressionResult with MLE coefficients and log-sigma.
    """
    _validate_df(data, y)
    if isinstance(x, str):
        x = [x]
    _validate_df(data, *x)
    df = data[[y, *x]].dropna()
    Y = df[y].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(df)), df[x].to_numpy(dtype=float)])
    n, p = X.shape

    censored = lower >= Y

    def neg_loglik(params):
        beta = params[:p]
        log_sigma = params[p]
        sigma = np.exp(log_sigma)
        mu = X @ beta
        z = (Y - mu) / sigma
        ll = np.where(
            censored,
            stats.norm.logcdf((lower - mu) / sigma),
            -log_sigma + stats.norm.logpdf(z),
        )
        return -np.sum(ll)

    beta0 = np.linalg.lstsq(X, Y, rcond=None)[0]
    sigma0 = np.log(np.std(Y - X @ beta0) + 1e-8)
    res = optimize.minimize(neg_loglik, np.append(beta0, sigma0), method="Nelder-Mead")
    beta = res.x[:p]
    sigma = np.exp(res.x[p])

    names = ["(Intercept)", *x]
    coefficients = {nm: float(b) for nm, b in zip(names, beta)}

    return RegressionResult(
        method="Tobit",
        coefficients=coefficients,
        se={nm: float("nan") for nm in names},
        p_values={nm: float("nan") for nm in names},
        residuals=Y - X @ beta,
        fitted=X @ beta,
        n=n,
        k=p - 1,
        extra={"lower": lower, "sigma": sigma, "n_censored": int(np.sum(censored))},
    )


tobit = tobit_model


def cheatsheet() -> str:
    return "tobit_model({}) -> Tobit censored regression. 'Your eyes can deceive you. -- Ob"
