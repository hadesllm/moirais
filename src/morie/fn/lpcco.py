# morie.fn -- function file (hadesllm/morie)
"""Linear Prediction Coding coefficients."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Judge me by my size, do you?"


def lpc_coefficients_fn(x: np.ndarray, order: int = 10) -> DescriptiveResult:
    """Compute Linear Prediction Coding (LPC) coefficients via autocorrelation method.

    LPC minimises the mean-squared prediction error by solving the
    Yule-Walker equations through Levinson-Durbin recursion.

    :param x: 1-D input signal.
    :param order: LPC order (default 10).
    :return: DescriptiveResult with LPC coefficients and prediction error variance.
    """
    from morie._armodel import ar_yule_walker

    x = np.asarray(x, dtype=float).ravel()
    a, sigma2 = ar_yule_walker(x, order=order)
    return DescriptiveResult(
        name="lpc_coefficients",
        value=float(sigma2),
        extra={"coefficients": a, "sigma2": float(sigma2), "order": order},
    )


lpcco = lpc_coefficients_fn


def cheatsheet() -> str:
    return "lpc_coefficients_fn({}) -> Linear Prediction Coding coefficients."
