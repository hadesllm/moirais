# morie.fn — function file (hadesllm/morie)
"""Correlation dimension estimation.

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 15.
"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

__all__ = ['corrn']
def corrn(
    x: np.ndarray,
    *,
    m: int = 3,
    tau: int = 1,
    r_min: float | None = None,
    r_max: float | None = None,
    n_r: int = 30,
) -> DescriptiveResult:
    """Estimate the correlation dimension (Grassberger-Procaccia).

    Parameters
    ----------
    x : array-like
        1-D time series.
    m : int
        Embedding dimension.
    tau : int
        Time delay (samples).
    r_min : float or None
        Minimum radius (default auto).
    r_max : float or None
        Maximum radius (default auto).
    n_r : int
        Number of radius values.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    N = n - (m - 1) * tau
    if N < 10:
        raise ValueError("Signal too short.")

    Y = np.array([x[i:i + (m - 1) * tau + 1:tau] for i in range(N)])

    dists = []
    for i in range(N):
        for j in range(i + 1, N):
            dists.append(np.max(np.abs(Y[i] - Y[j])))
    dists = np.array(dists)
    dists = dists[dists > 0]

    if len(dists) == 0:
        return DescriptiveResult(name="corrn", value=0.0,
                                 extra={"d2": 0.0, "log_r": [], "log_C": []})

    if r_min is None:
        r_min = np.percentile(dists, 1)
    if r_max is None:
        r_max = np.percentile(dists, 99)

    radii = np.logspace(np.log10(max(r_min, 1e-10)), np.log10(r_max), n_r)
    C = np.array([np.mean(dists <= r) for r in radii])

    valid = C > 0
    log_r = np.log(radii[valid])
    log_C = np.log(C[valid])

    if len(log_r) >= 4:
        mid = len(log_r) // 4
        end = 3 * len(log_r) // 4
        coeffs = np.polyfit(log_r[mid:end], log_C[mid:end], 1)
        d2 = float(coeffs[0])
    else:
        d2 = 0.0

    return DescriptiveResult(
        name="corrn",
        value=d2,
        extra={"d2": d2, "log_r": log_r, "log_C": log_C, "m": m, "tau": tau},
    )


def cheatsheet() -> str:
    return "corrn({}) -> Correlation dimension."
