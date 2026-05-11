# morie.fn — function file (hadesllm/morie)
"""Waste no more time arguing what a good person should be. Be one. — Marcus Aurelius"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def lagrange_interp(
    x_known: np.ndarray,
    y_known: np.ndarray,
    x_eval: np.ndarray,
) -> DescriptiveResult:
    """
    Lagrange polynomial interpolation.

    .. math::

        P(x) = \\sum_{i=0}^{n} y_i \\prod_{j \\neq i}
               \\frac{x - x_j}{x_i - x_j}

    :param x_known: Known x-coordinates (must be distinct).
    :param y_known: Known y-values at x_known.
    :param x_eval: Points at which to evaluate the interpolant.
    :return: DescriptiveResult with interpolated values.
    :raises ValueError: If x_known contains duplicates or lengths differ.

    References
    ----------
    Burden, R. L., & Faires, J. D. (2011). *Numerical Analysis*.
    9th ed. Brooks/Cole. Ch. 3.
    """
    xk = np.asarray(x_known, dtype=np.float64)
    yk = np.asarray(y_known, dtype=np.float64)
    xe = np.asarray(x_eval, dtype=np.float64)

    if len(xk) != len(yk):
        raise ValueError("x_known and y_known must have equal length.")
    if len(np.unique(xk)) != len(xk):
        raise ValueError("x_known must contain distinct values.")

    n = len(xk)
    xe_flat = xe.ravel()
    result = np.zeros_like(xe_flat)

    for i in range(n):
        basis = np.ones_like(xe_flat)
        for j in range(n):
            if i != j:
                basis *= (xe_flat - xk[j]) / (xk[i] - xk[j])
        result += yk[i] * basis

    return DescriptiveResult(
        name="Lagrange Interpolation",
        value=float(result[0]) if len(result) == 1 else float(np.mean(result)),
        extra={
            "y_eval": result.reshape(xe.shape) if xe.ndim > 0 else result,
            "degree": n - 1,
            "n_points": n,
        },
    )


short = lagrange_interp


def cheatsheet() -> str:
    return "lagrange_interp({}) -> Lagrange interpolation. 'Your eyes can deceive you. Don't tr"
