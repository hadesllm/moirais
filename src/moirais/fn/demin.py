# moirais.fn — function file (hadesllm/moirais)
"""Truth comes out of error more readily than out of confusion. — Francis Bacon"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def deming_regression(
    x: np.ndarray,
    y: np.ndarray,
    lambda_ratio: float = 1.0,
) -> DescriptiveResult:
    """
    Deming regression for errors-in-variables.

    Accounts for measurement error in both *x* and *y*. When
    ``lambda_ratio = var(epsilon_x) / var(epsilon_y) = 1``, this
    reduces to orthogonal regression.

    :param x: Predictor measurements (1-D).
    :param y: Response measurements (1-D), same length as *x*.
    :param lambda_ratio: Ratio of error variances. Default 1.0.
    :return: DescriptiveResult with slope as value.
    :raises ValueError: If inputs are invalid.

    References
    ----------
    Deming, W. E. (1943). Statistical Adjustment of Data. Wiley.
    Linnet, K. (1993). Evaluation of regression procedures for methods
    comparison studies. Clinical Chemistry, 39(3), 424--432.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or x.size < 3:
        raise ValueError("x and y must be 1-D arrays of equal length >= 3.")
    if lambda_ratio <= 0:
        raise ValueError(f"lambda_ratio must be > 0, got {lambda_ratio}.")

    n = len(x)
    xm, ym = np.mean(x), np.mean(y)
    sxx = np.sum((x - xm) ** 2) / (n - 1)
    syy = np.sum((y - ym) ** 2) / (n - 1)
    sxy = np.sum((x - xm) * (y - ym)) / (n - 1)

    diff = syy - lambda_ratio * sxx
    slope = (diff + np.sqrt(diff**2 + 4 * lambda_ratio * sxy**2)) / (2 * sxy)
    intercept = ym - slope * xm

    fitted = intercept + slope * x
    residuals = y - fitted

    return DescriptiveResult(
        name="Deming Regression",
        value=float(np.round(slope, 6)),
        extra={
            "slope": float(np.round(slope, 6)),
            "intercept": float(np.round(intercept, 6)),
            "lambda_ratio": lambda_ratio,
            "fitted": fitted,
            "residuals": residuals,
            "n": n,
        },
    )


demin = deming_regression


def cheatsheet() -> str:
    return "deming_regression({}) -> Deming (errors-in-variables) regression. 'The circle is now "
