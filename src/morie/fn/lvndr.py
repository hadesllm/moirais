# morie.fn — function file (hadesllm/morie)
"""Levinson-Durbin recursion for AR coefficient estimation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def levinson_durbin_fn(x: np.ndarray, order: int = 10) -> DescriptiveResult:
    """Solve Levinson-Durbin recursion from signal autocorrelation.

    :param x: 1-D input signal.
    :param order: AR model order (default 10).
    :return: DescriptiveResult with AR coefficients and prediction error in extra.
    """
    from morie._armodel import levinson_durbin

    x = np.asarray(x, dtype=float).ravel()
    r = np.array([np.correlate(x, x, mode="full")[len(x) - 1 - k] for k in range(order + 1)])
    a, e = levinson_durbin(r, order)
    return DescriptiveResult(
        name="levinson_durbin",
        value=None,
        extra={"coefficients": a, "prediction_error": float(e)},
    )


lvndr = levinson_durbin_fn


def cheatsheet() -> str:
    return "levinson_durbin_fn({}) -> Levinson-Durbin recursion for AR coefficient estimation."
