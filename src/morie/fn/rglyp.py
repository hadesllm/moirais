# morie.fn — function file (hadesllm/morie)
"""Largest Lyapunov exponent (Rosenstein) — Rangayyan Ch 7."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult, with_describe_pointer

__all__ = ["rangayyan_lyapunov"]


def rangayyan_lyapunov(x, m=3, tau=1, max_t=None, theiler=10):
    """Largest Lyapunov exponent via Rosenstein's algorithm.

    1. Delay-embed dimension ``m``, lag ``tau``.
    2. Nearest neighbour search with Theiler-window exclusion.
    3. Mean ``⟨ln d(t)⟩`` vs forward step ``t``.
    4. λ₁ = slope of the linear (early-growth) region.

    Parameters
    ----------
    x : array-like
    m : int
    tau : int
    max_t : int, optional
    theiler : int

    Returns
    -------
    RichResult with keys ``lyapunov``, ``divergence_curve``, ``t``.

    References
    ----------
    Rosenstein et al. (1993), Physica D 65:117. Rangayyan Ch 7.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    M = N - (m - 1) * tau
    if M < 10:
        raise ValueError("Series too short for embedding.")
    Y = np.empty((M, m))
    for i in range(m):
        Y[:, i] = x[i * tau:i * tau + M]
    if max_t is None:
        max_t = min(M // 4, 100)
    d = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=2)
    iv = np.arange(M)
    mask = np.abs(iv[:, None] - iv[None, :]) <= theiler
    d = np.where(mask, np.inf, d)
    nn = np.argmin(d, axis=1)
    div = np.full(max_t, np.nan)
    for t in range(max_t):
        ok = (iv + t < M) & (nn + t < M)
        if not ok.any():
            continue
        dij = np.linalg.norm(Y[iv[ok] + t] - Y[nn[ok] + t], axis=1)
        dij = dij[dij > 0]
        if dij.size:
            div[t] = float(np.mean(np.log(dij)))
    ts = np.where(np.isfinite(div))[0]
    if ts.size < 3:
        lam = float("nan")
    else:
        half = max(3, ts.size // 2)
        slope, _ = np.polyfit(ts[:half], div[ts[:half]], 1)
        lam = float(slope)
    res = RichResult(
        title="Largest Lyapunov exponent (Rosenstein)",
        summary_lines=[("m", m), ("τ", tau),
                       ("Theiler", theiler), ("λ₁", lam)],
        interpretation=f"λ₁ = {lam:.4g}. >0 chaotic, ~0 marginal, <0 stable.",
        payload={"lyapunov": lam, "divergence_curve": div, "t": np.arange(max_t)},
    )
    return with_describe_pointer(res, "rglyp")


# CANONICAL TEST
# >>> rng = np.random.default_rng(0)
# >>> r = rangayyan_lyapunov(rng.standard_normal(200), m=3, tau=1, max_t=20)
# >>> "lyapunov" in r
# True


def cheatsheet():
    return "rglyp: largest Lyapunov exponent (Rosenstein) — Rangayyan Ch 7"
