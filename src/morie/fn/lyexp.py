# morie.fn -- function file (hadesllm/morie)
"""Lyapunov exponent estimation (largest).

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 15.
"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

__all__ = ['lyexp']

_QUOTE = "That which does not kill us makes us stronger. -- Friedrich Nietzsche"


def lyexp(
    x: np.ndarray,
    *,
    m: int = 3,
    tau: int = 1,
    dt: float = 1.0,
    min_tsep: int | None = None,
    max_iter: int | None = None,
) -> DescriptiveResult:
    """Estimate the largest Lyapunov exponent via Rosenstein's method.

    Parameters
    ----------
    x : array-like
        1-D time series.
    m : int
        Embedding dimension.
    tau : int
        Time delay (samples).
    dt : float
        Sampling period.
    min_tsep : int or None
        Minimum temporal separation for nearest neighbours
        (default m * tau).
    max_iter : int or None
        Maximum iterations for divergence curve (default N//4).

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    N = n - (m - 1) * tau
    if N < 10:
        raise ValueError("Signal too short for given m and tau.")

    if min_tsep is None:
        min_tsep = m * tau
    if max_iter is None:
        max_iter = N // 4

    Y = np.array([x[i:i + (m - 1) * tau + 1:tau] for i in range(N)])

    nn = np.zeros(N, dtype=int)
    for i in range(N):
        min_dist = np.inf
        for j in range(N):
            if abs(i - j) < min_tsep:
                continue
            d = np.max(np.abs(Y[i] - Y[j]))
            if d < min_dist:
                min_dist = d
                nn[i] = j

    divergence = np.zeros(max_iter)
    counts = np.zeros(max_iter)
    for i in range(N):
        j = nn[i]
        for k in range(max_iter):
            ii = i + k
            jj = j + k
            if ii >= N or jj >= N:
                break
            d = np.max(np.abs(Y[ii] - Y[jj]))
            if d > 0:
                divergence[k] += np.log(d)
                counts[k] += 1

    valid = counts > 0
    divergence[valid] /= counts[valid]

    t_axis = np.arange(max_iter) * dt
    valid_idx = np.where(valid)[0]
    if len(valid_idx) >= 2:
        fit_end = min(len(valid_idx), max(10, max_iter // 4))
        idx = valid_idx[:fit_end]
        coeffs = np.polyfit(t_axis[idx], divergence[idx], 1)
        lam = float(coeffs[0])
    else:
        lam = 0.0

    return DescriptiveResult(
        name="lyexp",
        value=lam,
        extra={
            "lyapunov_exponent": lam,
            "divergence_curve": divergence,
            "t_axis": t_axis,
            "m": m,
            "tau": tau,
        },
    )


def cheatsheet() -> str:
    return "lyexp({}) -> Largest Lyapunov exponent."
