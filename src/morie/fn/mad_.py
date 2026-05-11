# morie.fn — function file (hadesllm/morie)
"""Median Absolute Deviation (MAD) statistic."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def median_abs_deviation(
    x: np.ndarray,
    *,
    constant: float = 1.4826,
) -> ESRes:
    """MAD — robust scale estimator.

    MAD = constant * median(|x_i - median(x)|).
    Default constant 1.4826 makes MAD consistent for the normal.

    Parameters
    ----------
    x : array-like
    constant : float
        Consistency constant.

    Returns
    -------
    ESRes
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 1:
        raise ValueError("Need >= 1 finite observation.")

    med = np.median(x)
    mad_raw = np.median(np.abs(x - med))
    mad_scaled = constant * mad_raw

    return ESRes(
        measure="mad",
        estimate=float(mad_scaled),
        n=len(x),
        extra={"median": float(med), "mad_raw": float(mad_raw), "constant": constant},
    )


mad_ = median_abs_deviation


def cheatsheet() -> str:
    return "median_abs_deviation({}) -> Median Absolute Deviation (MAD) statistic."
