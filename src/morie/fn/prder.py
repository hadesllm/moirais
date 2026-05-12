# morie.fn -- function file (hadesllm/morie)
"""Forward linear prediction error."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Great, kid. Don't get cocky."


def prediction_error_fn(x: np.ndarray, ar_coeffs: np.ndarray) -> DescriptiveResult:
    r"""Compute forward linear prediction error for signal *x* given AR coefficients.

    .. math::

        e(n) = x(n) - \\sum_{k=1}^{p} a_k \\, x(n-k)

    :param x: 1-D input signal.
    :param ar_coeffs: AR coefficients [a1, ..., ap].
    :return: DescriptiveResult with prediction error signal and MSE.
    """
    x = np.asarray(x, dtype=float).ravel()
    ar_coeffs = np.asarray(ar_coeffs, dtype=float).ravel()
    p = len(ar_coeffs)
    n = len(x)
    error = np.zeros(n)
    for i in range(p, n):
        pred = np.dot(ar_coeffs, x[i - p : i][::-1])
        error[i] = x[i] - pred
    mse = float(np.mean(error[p:] ** 2))
    return DescriptiveResult(
        name="prediction_error",
        value=mse,
        extra={"error": error, "mse": mse, "order": p},
    )


prder = prediction_error_fn


def cheatsheet() -> str:
    return "prediction_error_fn({}) -> Forward linear prediction error."
