# morie.fn -- function file (hadesllm/morie)
"""Quantile regression. 'The greatest teacher, failure is.'"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize

from ._containers import RegressionResult
from ._helpers import _validate_df


def quantile_regression(
    data: pd.DataFrame,
    *,
    y: str = "y",
    x: list[str] | str = "x",
    tau: float = 0.5,
) -> RegressionResult:
    """Quantile regression via linear programming (Nelder-Mead).

    :param data: DataFrame with outcome and predictor columns.
    :param y: Name of the outcome column.
    :param x: Predictor column name(s).
    :param tau: Quantile to estimate (0 < tau < 1). Default 0.5 = median.
    :return: RegressionResult with quantile coefficients.
    """
    _validate_df(data, y)
    if isinstance(x, str):
        x = [x]
    _validate_df(data, *x)
    df = data[[y, *x]].dropna()
    Y = df[y].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(df)), df[x].to_numpy(dtype=float)])
    n, p = X.shape

    def check_loss(beta):
        r = Y - X @ beta
        return np.sum(np.where(r >= 0, tau * r, (tau - 1) * r))

    beta0 = np.linalg.lstsq(X, Y, rcond=None)[0]
    res = optimize.minimize(check_loss, beta0, method="Nelder-Mead")
    beta = res.x

    fitted = X @ beta
    resid = Y - fitted

    names = ["(Intercept)", *x]
    coefficients = {nm: float(b) for nm, b in zip(names, beta)}

    return RegressionResult(
        method=f"Quantile (tau={tau})",
        coefficients=coefficients,
        se={nm: float("nan") for nm in names},
        p_values={nm: float("nan") for nm in names},
        residuals=resid,
        fitted=fitted,
        n=n,
        k=p - 1,
        extra={"tau": tau},
    )


qreg = quantile_regression


def cheatsheet() -> str:
    return "quantile_regression({}) -> Quantile regression. 'The greatest teacher, failure is.' -- "
