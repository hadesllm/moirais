# morie.fn -- function file (hadesllm/morie)
"""MAPE (Mean Absolute Percentage Error) forecast accuracy."""

import numpy as np

from ._containers import DescriptiveResult


def mape(actual: np.ndarray, forecast: np.ndarray) -> DescriptiveResult:
    r"""
    Mean Absolute Percentage Error.

    .. math::

        \\text{MAPE} = \\frac{100}{n} \\sum_{t=1}^{n}
        \\left| \\frac{y_t - \\hat{y}_t}{y_t} \\right|

    Observations where actual == 0 are excluded.

    :param actual: 1-D array of actual values.
    :param forecast: 1-D array of forecast values.
    :return: DescriptiveResult with MAPE percentage.
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
    mask = a != 0
    if not np.any(mask):
        raise ValueError("All actual values are zero; MAPE undefined.")
    ape = np.abs((a[mask] - f[mask]) / a[mask]) * 100
    val = float(np.mean(ape))
    return DescriptiveResult(
        name="mape",
        value=val,
        extra={
            "mape": val,
            "median_ape": float(np.median(ape)),
            "n_used": int(np.sum(mask)),
            "n_total": len(a),
        },
    )


mapef = mape


def cheatsheet() -> str:
    return "mape({}) -> Mean Absolute Percentage Error."
