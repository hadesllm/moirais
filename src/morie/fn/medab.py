# morie.fn -- function file (hadesllm/morie)
"""Median absolute deviation (MAD)."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def median_abs_dev(
    x,
    *,
    scale: float = 1.4826,
) -> ESRes:
    """Compute the median absolute deviation.

    Parameters
    ----------
    x : array-like
        Observations.
    scale : float
        Consistency constant for normal data (default 1.4826).

    Returns
    -------
    ESRes
    """
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 1:
        raise ValueError("Need at least 1 finite observation.")
    med = np.median(a)
    raw_mad = float(np.median(np.abs(a - med)))
    return ESRes(
        measure="mad",
        estimate=raw_mad * scale,
        n=len(a),
        extra={"raw_mad": raw_mad, "scale": scale, "median": float(med)},
    )


medab = median_abs_dev


def cheatsheet() -> str:
    return "median_abs_dev(x) -> Median absolute deviation."
