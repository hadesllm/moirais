# morie.fn — function file (hadesllm/morie)
"""Higuchi fractal dimension — Rangayyan Ch 7."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult, with_describe_pointer

__all__ = ["rangayyan_higuchi_fd"]


def rangayyan_higuchi_fd(x, kmax=10):
    """Higuchi (1988) fractal dimension.

    L_m(k) = (N-1)/(k² ⌊(N-m)/k⌋) Σ |x[m+ik] − x[m+(i-1)k]|;
    L(k) = mean_m L_m(k); HFD = slope of log L(k) vs log(1/k).

    Parameters
    ----------
    x : array-like
    kmax : int
        Maximum time lag.

    Returns
    -------
    RichResult with keys ``HFD``, ``log_L``, ``log_inv_k``, ``kmax``.

    References
    ----------
    Higuchi (1988), Physica D 31:277. Rangayyan Ch 7.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    if N < 4 or kmax < 2:
        raise ValueError("Need len(x) >= 4 and kmax >= 2.")
    kmax = int(min(kmax, N // 2))
    L = np.empty(kmax)
    for k in range(1, kmax + 1):
        lk = []
        for m in range(k):
            idx = np.arange(m, N, k)
            if idx.size < 2:
                continue
            diffs = np.sum(np.abs(np.diff(x[idx])))
            norm = (N - 1) / (k * np.floor((N - m) / k))
            lk.append((diffs / k) * norm)
        L[k - 1] = np.mean(lk) if lk else np.nan
    ks = np.arange(1, kmax + 1)
    log_L = np.log(L)
    log_inv_k = np.log(1.0 / ks)
    slope, intercept = np.polyfit(log_inv_k, log_L, 1)
    res = RichResult(
        title="Higuchi fractal dimension",
        summary_lines=[("HFD", float(slope)), ("kmax", kmax), ("N", N)],
        interpretation=f"HFD = {slope:.4g}. ~1 smooth, ~2 rough.",
        payload={"HFD": float(slope), "intercept": float(intercept),
                 "log_L": log_L, "log_inv_k": log_inv_k, "kmax": kmax},
    )
    return with_describe_pointer(res, "rghfd")


# CANONICAL TEST
# >>> rng = np.random.default_rng(0)
# >>> r = rangayyan_higuchi_fd(rng.standard_normal(500), kmax=8)
# >>> 1.0 <= r["HFD"] <= 2.5
# True


def cheatsheet():
    return "rghfd: Higuchi fractal dimension — Rangayyan Ch 7"
