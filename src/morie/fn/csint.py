# morie.fn — function file (hadesllm/morie)
"""Natural cubic spline interpolation. 'Stay on target.' -- Gold Five"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def cubic_spline_interp(
    x_known: np.ndarray,
    y_known: np.ndarray,
    x_eval: np.ndarray,
) -> DescriptiveResult:
    """
    Natural cubic spline interpolation (S''=0 at endpoints).

    Solves the tridiagonal system for second derivatives and evaluates
    the piecewise cubic polynomial.

    :param x_known: Known x-coordinates (sorted, distinct).
    :param y_known: Known y-values.
    :param x_eval: Evaluation points.
    :return: DescriptiveResult with interpolated values and coefficients.
    :raises ValueError: If fewer than 2 points or x not sorted.

    References
    ----------
    De Boor, C. (1978). *A Practical Guide to Splines*. Springer.
    """
    xk = np.asarray(x_known, dtype=np.float64)
    yk = np.asarray(y_known, dtype=np.float64)
    xe = np.asarray(x_eval, dtype=np.float64)

    if len(xk) != len(yk):
        raise ValueError("x_known and y_known must have equal length.")
    if len(xk) < 2:
        raise ValueError("Need at least 2 data points.")
    if not np.all(np.diff(xk) > 0):
        raise ValueError("x_known must be strictly increasing.")

    n = len(xk) - 1
    h = np.diff(xk)

    A = np.zeros(n + 1)
    if n > 1:
        rhs = np.zeros(n - 1)
        for i in range(1, n):
            rhs[i - 1] = 3.0 * ((yk[i + 1] - yk[i]) / h[i] - (yk[i] - yk[i - 1]) / h[i - 1])

        tri = np.zeros((n - 1, n - 1))
        for i in range(n - 1):
            tri[i, i] = 2.0 * (h[i] + h[i + 1])
            if i > 0:
                tri[i, i - 1] = h[i]
            if i < n - 2:
                tri[i, i + 1] = h[i + 1]

        A[1:n] = np.linalg.solve(tri, rhs)

    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    for i in range(n):
        b[i] = (yk[i + 1] - yk[i]) / h[i] - h[i] * (2 * A[i] + A[i + 1]) / 3.0
        c[i] = A[i]
        d[i] = (A[i + 1] - A[i]) / (3.0 * h[i])

    xe_flat = xe.ravel()
    result = np.zeros_like(xe_flat)
    for idx, x in enumerate(xe_flat):
        seg = min(max(np.searchsorted(xk, x) - 1, 0), n - 1)
        dx = x - xk[seg]
        result[idx] = yk[seg] + b[seg] * dx + c[seg] * dx**2 + d[seg] * dx**3

    return DescriptiveResult(
        name="Natural Cubic Spline",
        value=float(result[0]) if len(result) == 1 else float(np.mean(result)),
        extra={
            "y_eval": result.reshape(xe.shape) if xe.ndim > 0 else result,
            "n_segments": n,
            "coefficients_b": b,
            "coefficients_c": c,
            "coefficients_d": d,
        },
    )


short = cubic_spline_interp


def cheatsheet() -> str:
    return "cubic_spline_interp({}) -> Natural cubic spline interpolation. 'Stay on target.' -- Gol"
