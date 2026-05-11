# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Backward linear prediction error."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Let the Wookiee win."


def backward_prediction_fn(x: np.ndarray, ar_coeffs: np.ndarray) -> DescriptiveResult:
    """Compute backward linear prediction error.

    .. math::

        e_b(n) = x(n) - \\sum_{k=1}^{p} a_k \\, x(n+k)

    :param x: 1-D input signal.
    :param ar_coeffs: AR coefficients [a1, ..., ap].
    :return: DescriptiveResult with backward prediction error signal and MSE.
    """
    x = np.asarray(x, dtype=float).ravel()
    ar_coeffs = np.asarray(ar_coeffs, dtype=float).ravel()
    p = len(ar_coeffs)
    n = len(x)
    error = np.zeros(n)
    for i in range(0, n - p):
        pred = np.dot(ar_coeffs, x[i + 1 : i + p + 1])
        error[i] = x[i] - pred
    valid = error[: n - p]
    mse = float(np.mean(valid**2)) if len(valid) > 0 else 0.0
    return DescriptiveResult(
        name="backward_prediction",
        value=mse,
        extra={"error": error, "mse": mse, "order": p},
    )


bkprd = backward_prediction_fn


def cheatsheet() -> str:
    return "backward_prediction_fn({}) -> Backward linear prediction error."
