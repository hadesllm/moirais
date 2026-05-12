# morie.fn -- function file (hadesllm/morie)
"""Negative binomial regression. 'Always in motion is the future.'"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, special

from ._containers import RegressionResult
from ._helpers import _validate_df


def negative_binomial_reg(
    data: pd.DataFrame,
    *,
    y: str = "y",
    x: list[str] | str = "x",
) -> RegressionResult:
    """Negative binomial (NB2) regression via MLE.

    Useful when count data exhibit overdispersion (variance > mean)
    relative to Poisson.  The NB2 parameterisation uses
    Var(Y) = mu + alpha * mu^2.

    :param data: DataFrame with count outcome and predictors.
    :param y: Name of the count outcome column.
    :param x: Predictor column name(s).
    :return: RegressionResult with NB coefficients.
    """
    _validate_df(data, y)
    if isinstance(x, str):
        x = [x]
    _validate_df(data, *x)
    df = data[[y, *x]].dropna()
    Y = df[y].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(df)), df[x].to_numpy(dtype=float)])
    n, p = X.shape

    def neg_loglik(params):
        beta = params[:p]
        log_alpha = params[p]
        alpha = np.exp(log_alpha)
        mu = np.exp(X @ beta)
        r = 1.0 / alpha
        ll = (
            special.gammaln(Y + r)
            - special.gammaln(r)
            - special.gammaln(Y + 1)
            + r * np.log(r / (r + mu))
            + Y * np.log(mu / (r + mu))
        )
        return -np.sum(ll)

    beta0 = np.zeros(p)
    beta0[0] = np.log(np.mean(Y) + 1e-8)
    init = np.append(beta0, 0.0)
    res = optimize.minimize(neg_loglik, init, method="Nelder-Mead", options={"maxiter": 5000})
    beta = res.x[:p]
    alpha = np.exp(res.x[p])
    mu = np.exp(X @ beta)

    names = ["(Intercept)", *x]
    return RegressionResult(
        method="Negative Binomial (NB2)",
        coefficients={nm: float(b) for nm, b in zip(names, beta)},
        se={nm: float("nan") for nm in names},
        p_values={nm: float("nan") for nm in names},
        fitted=mu,
        residuals=Y - mu,
        n=n,
        k=p - 1,
        extra={"alpha": alpha, "log_likelihood": -res.fun},
    )


negbn = negative_binomial_reg


def cheatsheet() -> str:
    return "negative_binomial_reg({}) -> Negative binomial regression. 'Always in motion is the futur"
