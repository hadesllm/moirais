"""System identification via least squares."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "It's a trap!"


def system_identify_fn(
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    order: int = 4,
) -> DescriptiveResult:
    r"""Identify a linear system from input/output data via least squares.

    Estimates the FIR coefficients :math:`h(k)` that minimise
    :math:`\\|y - X h\\|^2` where *X* is the convolution matrix.

    :param input_signal: 1-D input signal.
    :param output_signal: 1-D output signal (same length as input).
    :param order: System order / number of FIR taps (default 4).
    :return: DescriptiveResult with estimated impulse response and fit error.
    """
    u = np.asarray(input_signal, dtype=float).ravel()
    y = np.asarray(output_signal, dtype=float).ravel()
    n = min(len(u), len(y))
    u, y = u[:n], y[:n]
    X = np.zeros((n, order))
    for i in range(n):
        for j in range(order):
            if i - j >= 0:
                X[i, j] = u[i - j]
    h = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ h
    residual = y - y_hat
    mse = float(np.mean(residual**2))
    return DescriptiveResult(
        name="system_identify",
        value=mse,
        extra={"impulse_response": h, "predicted": y_hat, "mse": mse, "order": order},
    )


sysid = system_identify_fn


def cheatsheet() -> str:
    return "system_identify_fn({}) -> System identification via least squares."
