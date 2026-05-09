"""Theil-Sen robust trend estimator."""

import numpy as np
from scipy import stats as sp_stats

from ._containers import ESRes


def theil_sen(x: np.ndarray, y: np.ndarray) -> ESRes:
    """
    Theil-Sen median slope estimator for robust linear trend.

    Computes the median of all pairwise slopes:

    .. math::

        \\hat{\\beta} = \\text{median}\\left\\{
        \\frac{y_j - y_i}{x_j - x_i} : i < j\\right\\}

    :param x: (n,) independent variable.
    :param y: (n,) dependent variable.
    :return: ESRes with slope estimate, intercept and CI in extra.
    :raises ValueError: If x and y have different lengths.

    References
    ----------
    Theil H (1950). A rank-invariant method of linear and polynomial
    regression analysis. Proceedings of the Royal Netherlands Academy
    of Sciences, 53, 386-392.

    Sen PK (1968). Estimates of the regression coefficient based on
    Kendall's tau. JASA, 63(324), 1379-1389.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    result = sp_stats.theilslopes(y, x)
    slope = float(result[0])
    intercept = float(result[1])
    lo_slope = float(result[2])
    hi_slope = float(result[3])
    return ESRes(
        measure="theil_sen",
        estimate=slope,
        ci_lower=lo_slope,
        ci_upper=hi_slope,
        n=len(x),
        extra={"intercept": intercept},
    )


theil = theil_sen


def cheatsheet() -> str:
    return "theil_sen({}) -> Theil-Sen robust trend estimator."
