# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Fit an allometric (power-law) scaling model :math:`y = a x^b`."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import RegressionResult


def allometric_regression(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    *,
    alpha: float = 0.05,
) -> RegressionResult:
    r"""Fit an allometric (power-law) scaling model :math:`y = a x^b`.

    Takes the log-log transform and fits OLS: :math:`\ln y = \ln a + b \ln x`.

    Parameters
    ----------
    x, y : array-like
        Positive predictor and response values.
    alpha : float
        Significance level for CIs.

    Returns
    -------
    RegressionResult
        Coefficients ``ln_a`` and ``b`` (the scaling exponent) with
        standard errors and p-values.  ``r_squared`` is from log-log fit.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if len(x_arr) != len(y_arr):
        raise ValueError("x and y must have same length")
    if np.any(x_arr <= 0) or np.any(y_arr <= 0):
        raise ValueError("All values must be positive for log transform")
    if len(x_arr) < 3:
        raise ValueError("Need at least 3 observations")

    lx = np.log(x_arr)
    ly = np.log(y_arr)
    n = len(lx)

    slope, intercept, r_value, p_value, std_err = stats.linregress(lx, ly)

    residuals = ly - (intercept + slope * lx)
    mse = float(np.sum(residuals**2) / (n - 2))
    se_intercept = float(np.sqrt(mse * (1.0 / n + np.mean(lx) ** 2 / np.sum((lx - np.mean(lx)) ** 2))))
    t_int = intercept / se_intercept if se_intercept > 0 else 0
    p_int = float(2 * stats.t.sf(abs(t_int), n - 2))

    return RegressionResult(
        method="allometric (log-log OLS)",
        coefficients={"ln_a": float(intercept), "b": float(slope)},
        se={"ln_a": se_intercept, "b": float(std_err)},
        p_values={"ln_a": p_int, "b": float(p_value)},
        r_squared=float(r_value**2),
        adj_r_squared=1 - (1 - r_value**2) * (n - 1) / (n - 2),
        residuals=residuals,
        fitted=np.exp(intercept + slope * lx),
        n=n,
        k=2,
        extra={
            "a": float(np.exp(intercept)),
            "b": float(slope),
            "scaling_law": f"y = {np.exp(intercept):.4f} * x^{slope:.4f}",
        },
    )


allreg = allometric_regression


def cheatsheet() -> str:
    return 'allometric_regression({}) -> Allometric scaling regression.'
