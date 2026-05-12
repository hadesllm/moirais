# morie.fn — function file (hadesllm/morie)
"""Correlation dimension (Grassberger-Procaccia) — Rangayyan Ch 7."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult, with_describe_pointer

__all__ = ["rangayyan_correlation_dimension"]


def rangayyan_correlation_dimension(x, m=3, tau=1, n_r=20):
    """Correlation dimension D₂ (Grassberger-Procaccia 1983).

    1. Delay-embed ``x`` to dimension ``m`` with lag ``tau``.
    2. Correlation sum::

           C(r) = (1/(M(M-1))) Σ_{i≠j} Θ(r - ||Y_i - Y_j||)

    3. D₂ = slope of log C(r) vs log r in the scaling region.

    Parameters
    ----------
    x : array-like
    m : int
        Embedding dimension.
    tau : int
        Embedding lag.
    n_r : int
        Number of radii.

    Returns
    -------
    RichResult with keys ``D2``, ``log_r``, ``log_C``, ``m``, ``tau``.

    References
    ----------
    Grassberger & Procaccia (1983), Physica D 9:189. Rangayyan Ch 7.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    M = N - (m - 1) * tau
    if M < 10:
        raise ValueError("Series too short for embedding.")
    Y = np.empty((M, m))
    for i in range(m):
        Y[:, i] = x[i * tau:i * tau + M]
    d = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=2)
    iu = np.triu_indices(M, k=1)
    dist = d[iu]
    if dist.size == 0:
        raise ValueError("No pairwise distances.")
    pos = dist[dist > 0]
    rmin = max(pos.min() if pos.size else 1e-12, 1e-12)
    rmax = dist.max()
    rs = np.logspace(np.log10(rmin), np.log10(rmax), n_r)
    C = np.array([np.mean(dist <= r) for r in rs])
    mask = (C > 0) & np.isfinite(C)
    log_r = np.log(rs[mask]); log_C = np.log(C[mask])
    if log_r.size < 3:
        D2 = float("nan")
    else:
        n = log_r.size
        lo = max(1, n // 5); hi = max(lo + 2, n - n // 5)
        slope, _ = np.polyfit(log_r[lo:hi], log_C[lo:hi], 1)
        D2 = float(slope)
    res = RichResult(
        title="Correlation dimension (Grassberger-Procaccia)",
        summary_lines=[("m", m), ("τ", tau), ("D₂", D2)],
        interpretation=f"D₂ = {D2:.4g}. Saturates with m for low-dim chaos.",
        payload={"D2": D2, "log_r": log_r, "log_C": log_C,
                 "m": m, "tau": tau},
    )
    return with_describe_pointer(res, "rgcrl")


# CANONICAL TEST
# >>> rng = np.random.default_rng(0)
# >>> r = rangayyan_correlation_dimension(rng.standard_normal(200), m=3, tau=1, n_r=15)
# >>> np.isfinite(r["D2"])
# True


def cheatsheet():
    return "rgcrl: correlation dimension D₂ — Rangayyan Ch 7"
