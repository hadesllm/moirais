# morie.fn — function file (hadesllm/morie)
"""The only true wisdom is in knowing you know nothing. — Socrates"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize

from ._containers import RegressionResult
from ._helpers import _validate_df


def huber_regression(
    data: pd.DataFrame,
    *,
    y: str = "y",
    x: list[str] | str = "x",
    delta: float = 1.345,
) -> RegressionResult:
    """Huber M-estimator: robust to outliers via Huber loss.

    :param data: DataFrame with outcome and predictor columns.
    :param y: Name of the outcome column.
    :param x: Predictor column name(s).
    :param delta: Huber threshold (default 1.345 for 95% efficiency at normal).
    :return: RegressionResult with robust coefficients.
    """
    _validate_df(data, y)
    if isinstance(x, str):
        x = [x]
    _validate_df(data, *x)
    df = data[[y, *x]].dropna()
    Y = df[y].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(df)), df[x].to_numpy(dtype=float)])
    n, p = X.shape

    def huber_loss(beta):
        r = Y - X @ beta
        return np.sum(np.where(np.abs(r) <= delta, 0.5 * r**2, delta * (np.abs(r) - 0.5 * delta)))

    beta0 = np.linalg.lstsq(X, Y, rcond=None)[0]
    res = optimize.minimize(huber_loss, beta0, method="L-BFGS-B")
    beta = res.x

    fitted = X @ beta
    resid = Y - fitted
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    names = ["(Intercept)", *x]
    coefficients = {nm: float(b) for nm, b in zip(names, beta)}

    return RegressionResult(
        method=f"Huber (delta={delta})",
        coefficients=coefficients,
        se={nm: float("nan") for nm in names},
        p_values={nm: float("nan") for nm in names},
        r_squared=r2,
        residuals=resid,
        fitted=fitted,
        n=n,
        k=p - 1,
    )


hux = huber_regression


def cheatsheet() -> str:
    return "huber_regression({}) -> Huber robust regression. 'Careful not to choke on your aspir"
