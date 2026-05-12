# morie.fn -- function file (hadesllm/morie)
"""Higuchi fractal dimension."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def higuchi_fd(
    x: np.ndarray,
    *,
    kmax: int = 10,
) -> DescriptiveResult:
    """Higuchi fractal dimension of a 1-D signal.

    Estimates fractal dimension by computing curve length L(k) for
    k = 1..kmax and fitting log(L) vs log(1/k). The slope is the
    fractal dimension D (1 <= D <= 2).

    :param x: 1-D input signal (length >= 2 * kmax).
    :param kmax: Maximum interval (default 10). Auto-reduced if too large.
    :return: DescriptiveResult with D in ``value``.

    References
    ----------
    Higuchi, T. (1988). Approach to an irregular time series on the basis
    of the fractal theory. *Physica D*, 31(2), 277-283.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    kmax = min(kmax, n // 2)
    if kmax < 1:
        return DescriptiveResult(name="higuchi_fd", value=float("nan"))

    lk = np.zeros(kmax)
    for k in range(1, kmax + 1):
        lengths = []
        for m in range(1, k + 1):
            idx = np.arange(m - 1, n, k)
            if len(idx) < 2:
                continue
            seg = x[idx]
            norm = (n - 1) / (k * len(seg) * k)
            length = np.sum(np.abs(np.diff(seg))) * norm
            lengths.append(length)
        lk[k - 1] = np.mean(lengths) if lengths else 0.0

    valid = lk > 0
    if np.sum(valid) < 2:
        return DescriptiveResult(name="higuchi_fd", value=float("nan"))

    ks = np.arange(1, kmax + 1, dtype=float)[valid]
    log_k = np.log(1.0 / ks)
    log_l = np.log(lk[valid])
    A = np.column_stack([log_k, np.ones(len(log_k))])
    slope = np.linalg.lstsq(A, log_l, rcond=None)[0][0]

    return DescriptiveResult(
        name="higuchi_fd",
        value=float(slope),
        extra={"kmax": kmax, "log_k": log_k, "log_l": log_l},
    )


hfd = higuchi_fd


def cheatsheet() -> str:
    return "higuchi_fd({}) -> Higuchi fractal dimension."
