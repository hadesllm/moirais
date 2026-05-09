# moirais.fn — function file (hadesllm/moirais)
"""System linearity test."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Truly wonderful the mind of a child is."


def linearity_test(x, y, **kwargs) -> DescriptiveResult:
    """Test system linearity via R-squared of a linear fit.

    Parameters
    ----------
    x : array-like
        Input values.
    y : array-like
        Output values.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    coeffs = np.polyfit(x, y, 1)
    y_hat = np.polyval(coeffs, x)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-15)
    linear = bool(r2 > 0.95)
    return DescriptiveResult(
        name="linearity_test",
        value=float(r2),
        extra={
            "r_squared": float(r2),
            "slope": float(coeffs[0]),
            "intercept": float(coeffs[1]),
            "linear": linear,
        },
    )


lnrty = linearity_test


def cheatsheet() -> str:
    return "linearity_test({}) -> System linearity test."
