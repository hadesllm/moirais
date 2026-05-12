# morie.fn -- function file (hadesllm/morie)
"""MA coefficient estimation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Wars not make one great."


def ma_estimate_fn(x: np.ndarray, order: int = 4) -> DescriptiveResult:
    """Estimate Moving Average (MA) model coefficients.

    Uses the innovations algorithm to estimate MA(q) parameters from
    the autocorrelation of *x*.

    :param x: 1-D input signal.
    :param order: MA model order q (default 4).
    :return: DescriptiveResult with MA coefficients and innovation variance.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    x_c = x - np.mean(x)
    r = np.correlate(x_c, x_c, mode="full")
    r = r[n - 1 :] / n
    q = order
    theta = np.zeros((q + 1, q + 1))
    v = np.zeros(q + 1)
    v[0] = r[0]
    for i in range(1, q + 1):
        for k in range(i):
            s = 0.0
            for j in range(k):
                s += theta[i - 1, i - 1 - j] * theta[k, k - j] * v[j]
            theta[i, i - 1 - k] = (r[i - k] - s) / v[k] if v[k] != 0 else 0.0
        v[i] = r[0]
        for j in range(i):
            v[i] -= theta[i, i - 1 - j] ** 2 * v[j]
        v[i] = max(v[i], 1e-20)
    ma_coeffs = theta[q, :q]
    sigma2 = float(v[q])
    return DescriptiveResult(
        name="ma_estimate",
        value=float(sigma2),
        extra={"ma_coeffs": ma_coeffs, "sigma2": sigma2, "order": q},
    )


maest = ma_estimate_fn


def cheatsheet() -> str:
    return "ma_estimate_fn({}) -> MA coefficient estimation."
