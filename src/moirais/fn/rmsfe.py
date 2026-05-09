# moirais.fn — function file (hadesllm/moirais)
"""RMSFE (Root Mean Square Forecast Error)."""

import numpy as np

from ._containers import DescriptiveResult


def rmsfe_calc(actual: np.ndarray, forecast: np.ndarray) -> DescriptiveResult:
    """
    Root Mean Square Forecast Error.

    .. math::

        \\text{RMSFE} = \\sqrt{\\frac{1}{n} \\sum_{t=1}^{n}
        (y_t - \\hat{y}_t)^2}

    :param actual: 1-D array of actual values.
    :param forecast: 1-D array of forecast values.
    :return: DescriptiveResult with RMSFE, MAE, bias.
    :raises ValueError: On length mismatch.

    References
    ----------
    Hyndman R.J. & Koehler A.B. (2006). Another look at measures of
    forecast accuracy. *IJF*, 22(4), 679-688.
    """
    a = np.asarray(actual, dtype=float).ravel()
    f = np.asarray(forecast, dtype=float).ravel()
    if len(a) != len(f):
        raise ValueError(f"Lengths differ: {len(a)} vs {len(f)}.")
    n = len(a)
    if n == 0:
        raise ValueError("Empty arrays.")
    errors = a - f
    val = float(np.sqrt(np.mean(errors ** 2)))
    return DescriptiveResult(
        name="rmsfe",
        value=val,
        extra={
            "rmsfe": val,
            "mae": float(np.mean(np.abs(errors))),
            "bias": float(np.mean(errors)),
            "n": n,
        },
    )


rmsfe = rmsfe_calc


def cheatsheet() -> str:
    return "rmsfe_calc({}) -> Root Mean Square Forecast Error."
