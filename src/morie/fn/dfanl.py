# morie.fn -- function file (hadesllm/morie)
"""Detrended fluctuation analysis (DFA).

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 15.
"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

__all__ = ['dfanl']

_QUOTE = "Knowledge is power. -- Francis Bacon"


def dfanl(
    x: np.ndarray,
    *,
    n_min: int = 4,
    n_max: int | None = None,
    n_scales: int = 20,
    order: int = 1,
) -> DescriptiveResult:
    """Detrended fluctuation analysis.

    Parameters
    ----------
    x : array-like
        1-D time series.
    n_min : int
        Minimum box size.
    n_max : int or None
        Maximum box size (default len(x)//4).
    n_scales : int
        Number of box sizes.
    order : int
        Polynomial order for detrending.

    Returns
    -------
    DescriptiveResult
        ``value`` is the DFA scaling exponent alpha.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = len(x)
    y = np.cumsum(x - np.mean(x))

    if n_max is None:
        n_max = N // 4
    n_max = max(n_max, n_min + 1)

    scales = np.unique(np.logspace(
        np.log10(n_min), np.log10(n_max), n_scales
    ).astype(int))
    scales = scales[scales >= n_min]

    fluct = np.zeros(len(scales))
    for si, s in enumerate(scales):
        n_seg = N // s
        if n_seg < 1:
            fluct[si] = np.nan
            continue
        rms_vals = []
        for v in range(n_seg):
            seg = y[v * s:(v + 1) * s]
            t = np.arange(s)
            p = np.polyfit(t, seg, order)
            trend = np.polyval(p, t)
            rms_vals.append(np.sqrt(np.mean((seg - trend) ** 2)))
        fluct[si] = np.mean(rms_vals)

    valid = ~np.isnan(fluct) & (fluct > 0)
    log_n = np.log(scales[valid].astype(float))
    log_f = np.log(fluct[valid])

    if len(log_n) >= 2:
        coeffs = np.polyfit(log_n, log_f, 1)
        alpha = float(coeffs[0])
    else:
        alpha = 0.0

    return DescriptiveResult(
        name="dfanl",
        value=alpha,
        extra={
            "alpha": alpha,
            "scales": scales,
            "fluctuations": fluct,
            "log_n": log_n,
            "log_f": log_f,
        },
    )


def cheatsheet() -> str:
    return "dfanl({}) -> Detrended fluctuation analysis."
