# morie.fn — function file (hadesllm/morie)
"""Detrended fluctuation analysis — Rangayyan Ch 7."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult, with_describe_pointer

__all__ = ["rangayyan_dfa"]


def rangayyan_dfa(x, scales=None, order=1):
    """DFA scaling exponent α (Peng et al. 1994).

    1. Y(k) = Σ_{i=1}^{k} (x_i − mean(x)).
    2. Partition Y into boxes of size ``n``; detrend each with a
       polynomial of order ``order``.
    3. F(n) = sqrt(mean residual variance across boxes).
    4. α = slope of log F(n) vs log n.

    Parameters
    ----------
    x : array-like
    scales : array-like of int, optional
        Box sizes; default geometric 4 .. N/4.
    order : int
        Detrending polynomial order (DFA-1 default).

    Returns
    -------
    RichResult with keys ``alpha``, ``scales``, ``F``, ``log_scales``, ``log_F``.

    References
    ----------
    Peng, C.-K. et al. (1994), Phys Rev E 49:1685. Rangayyan Ch 7.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    if N < 32:
        raise ValueError("DFA needs at least 32 samples.")
    if scales is None:
        log_n = np.linspace(np.log(4), np.log(max(8, N // 4)), 12)
        scales = np.unique(np.round(np.exp(log_n)).astype(int))
        scales = scales[scales >= 4]
    scales = np.asarray(scales, dtype=int)
    y = np.cumsum(x - x.mean())
    F = np.empty(scales.size, dtype=float)
    for j, n in enumerate(scales):
        nseg = N // n
        if nseg < 1:
            F[j] = np.nan
            continue
        rms = []
        for k in range(nseg):
            seg = y[k * n:(k + 1) * n]
            t = np.arange(n)
            p = np.polyfit(t, seg, order)
            trend = np.polyval(p, t)
            rms.append(np.mean((seg - trend) ** 2))
        F[j] = np.sqrt(np.mean(rms))
    mask = np.isfinite(F) & (F > 0)
    log_n = np.log(scales[mask])
    log_F = np.log(F[mask])
    alpha, intercept = np.polyfit(log_n, log_F, 1)
    res = RichResult(
        title="Detrended Fluctuation Analysis",
        summary_lines=[("α", float(alpha)), ("Order", int(order)),
                       ("Scales", len(scales))],
        interpretation=f"α = {alpha:.4g}. 0.5 random, 1 1/f, >1 persistent.",
        payload={"alpha": float(alpha), "scales": scales, "F": F,
                 "log_scales": log_n, "log_F": log_F},
    )
    return with_describe_pointer(res, "rgdfa")


# CANONICAL TEST
# >>> rng = np.random.default_rng(0)
# >>> r = rangayyan_dfa(rng.standard_normal(500))
# >>> 0.3 < r["alpha"] < 0.7
# True


def cheatsheet():
    return "rgdfa: detrended fluctuation analysis α — Rangayyan Ch 7"
