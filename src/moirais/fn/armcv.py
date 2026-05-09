# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""AR model estimation via modified covariance method."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Our greatest glory is not in never falling, but in rising every time we fall. — Confucius"


def ar_modified_cov_fn(x: np.ndarray, order: int = 4) -> DescriptiveResult:
    """Estimate AR coefficients via the modified covariance method.

    Averages forward and backward covariance matrices for reduced bias
    compared to the standard covariance method.

    :param x: 1-D input signal.
    :param order: AR model order (default 4).
    :return: DescriptiveResult with coefficients and residual variance.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if order >= n:
        raise ValueError("order must be less than signal length")
    R = np.zeros((order, order))
    r = np.zeros(order)
    for i in range(order):
        for j in range(order):
            fwd = 0.0
            bwd = 0.0
            for k in range(order, n):
                fwd += x[k - i - 1] * x[k - j - 1]
            for k in range(0, n - order):
                bwd += x[k + i + 1] * x[k + j + 1]
            R[i, j] = (fwd + bwd) / 2
        fwd_r = 0.0
        bwd_r = 0.0
        for k in range(order, n):
            fwd_r += x[k] * x[k - i - 1]
        for k in range(0, n - order):
            bwd_r += x[k] * x[k + i + 1]
        r[i] = (fwd_r + bwd_r) / 2
    try:
        a = np.linalg.solve(R, r)
    except np.linalg.LinAlgError:
        a = np.linalg.lstsq(R, r, rcond=None)[0]
    residual = x[order:].copy()
    for i in range(order):
        residual -= a[i] * x[order - i - 1 : n - i - 1]
    sigma2 = float(np.mean(residual**2))
    return DescriptiveResult(
        name="ar_modified_covariance",
        value=float(sigma2),
        extra={"coefficients": a, "sigma2": sigma2, "order": order},
    )


armcv = ar_modified_cov_fn


def cheatsheet() -> str:
    return "ar_modified_cov_fn({}) -> AR model estimation via modified covariance method."
