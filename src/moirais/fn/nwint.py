# moirais.fn — function file (hadesllm/moirais)
"""Newton divided-difference interpolation. 'The greatest teacher, failure is.'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def newton_interp(
    x_known: np.ndarray,
    y_known: np.ndarray,
    x_eval: np.ndarray,
) -> DescriptiveResult:
    """
    Newton's divided-difference interpolation.

    Builds the divided-difference table and evaluates the Newton form
    polynomial at given points.

    :param x_known: Known x-coordinates (must be distinct).
    :param y_known: Known y-values.
    :param x_eval: Evaluation points.
    :return: DescriptiveResult with interpolated values and coefficients.
    :raises ValueError: If arrays are incompatible.

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
    coeff = yk.copy()
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coeff[i] = (coeff[i] - coeff[i - 1]) / (xk[i] - xk[i - j])

    xe_flat = xe.ravel()
    result = np.full_like(xe_flat, coeff[-1])
    for i in range(n - 2, -1, -1):
        result = result * (xe_flat - xk[i]) + coeff[i]

    return DescriptiveResult(
        name="Newton Interpolation",
        value=float(result[0]) if len(result) == 1 else float(np.mean(result)),
        extra={
            "y_eval": result.reshape(xe.shape) if xe.ndim > 0 else result,
            "coefficients": coeff,
            "degree": n - 1,
        },
    )


short = newton_interp


def cheatsheet() -> str:
    return "newton_interp({}) -> Newton divided-difference interpolation. 'The greatest teach"
