"""Winsorized mean."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def winsorize(
    x: np.ndarray,
    *,
    trim: float = 0.1,
) -> ESRes:
    """Winsorized mean -- extreme values replaced by boundary percentiles.

    Parameters
    ----------
    x : array-like
    trim : float
        Fraction to winsorise from each tail (0 to 0.5).

    Returns
    -------
    ESRes
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        raise ValueError("Need >= 3 finite observations.")
    if not 0.0 <= trim < 0.5:
        raise ValueError("trim must be in [0, 0.5).")

    lo = np.percentile(x, 100 * trim)
    hi = np.percentile(x, 100 * (1 - trim))
    xw = np.clip(x, lo, hi)
    wm = float(np.mean(xw))
    se = float(np.std(xw, ddof=1) / np.sqrt(len(xw)))

    return ESRes(
        measure="winsorized_mean",
        estimate=wm,
        se=se,
        n=len(x),
        extra={"trim": trim, "lo_bound": float(lo), "hi_bound": float(hi)},
    )


winsz = winsorize


def cheatsheet() -> str:
    return "winsorize({}) -> Winsorized mean."
