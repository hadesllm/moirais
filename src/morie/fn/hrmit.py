# morie.fn — function file (hadesllm/morie)
"""All models are wrong, but some are useful. — George E. P. Box"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def hermite_interp(
    x_known: np.ndarray,
    y_known: np.ndarray,
    dy_known: np.ndarray,
    x_eval: np.ndarray,
) -> DescriptiveResult:
    """
    Hermite interpolation using both function values and derivatives.

    Constructs the divided-difference table for the osculating
    polynomial of degree 2n-1.

    :param x_known: Known x-coordinates (must be distinct).
    :param y_known: Known y-values.
    :param dy_known: Known derivative values at x_known.
    :param x_eval: Evaluation points.
    :return: DescriptiveResult with interpolated values.
    :raises ValueError: If array lengths are inconsistent.

    References
    ----------
    Burden, R. L., & Faires, J. D. (2011). *Numerical Analysis*.
    9th ed. Brooks/Cole. Sec. 3.4.
    """
    xk = np.asarray(x_known, dtype=np.float64)
    yk = np.asarray(y_known, dtype=np.float64)
    dk = np.asarray(dy_known, dtype=np.float64)
    xe = np.asarray(x_eval, dtype=np.float64)

    if not (len(xk) == len(yk) == len(dk)):
        raise ValueError("x_known, y_known, dy_known must have equal length.")
    if len(np.unique(xk)) != len(xk):
        raise ValueError("x_known must contain distinct values.")

    n = len(xk)
    m = 2 * n
    z = np.zeros(m)
    Q = np.zeros((m, m))

    for i in range(n):
        z[2 * i] = xk[i]
        z[2 * i + 1] = xk[i]
        Q[2 * i, 0] = yk[i]
        Q[2 * i + 1, 0] = yk[i]
        Q[2 * i + 1, 1] = dk[i]
        if i > 0:
            Q[2 * i, 1] = (Q[2 * i, 0] - Q[2 * i - 1, 0]) / (z[2 * i] - z[2 * i - 1])

    for j in range(2, m):
        for i in range(j, m):
            Q[i, j] = (Q[i, j - 1] - Q[i - 1, j - 1]) / (z[i] - z[i - j])

    coeff = np.array([Q[i, i] for i in range(m)])

    xe_flat = xe.ravel()
    result = np.full_like(xe_flat, coeff[-1])
    for i in range(m - 2, -1, -1):
        result = result * (xe_flat - z[i]) + coeff[i]

    return DescriptiveResult(
        name="Hermite Interpolation",
        value=float(result[0]) if len(result) == 1 else float(np.mean(result)),
        extra={
            "y_eval": result.reshape(xe.shape) if xe.ndim > 0 else result,
            "degree": m - 1,
            "n_points": n,
        },
    )


short = hermite_interp


def cheatsheet() -> str:
    return "hermite_interp({}) -> Hermite interpolation. 'Impressive. Most impressive.' -- Dar"
