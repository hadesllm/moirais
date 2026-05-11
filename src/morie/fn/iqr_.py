# morie.fn — function file (hadesllm/morie)
"""Interquartile range statistic."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def iqr_statistic(x: np.ndarray) -> ESRes:
    """Interquartile range (Q3 - Q1).

    Parameters
    ----------
    x : array-like

    Returns
    -------
    ESRes
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        raise ValueError("Need >= 4 finite observations.")

    q1 = float(np.percentile(x, 25))
    q3 = float(np.percentile(x, 75))
    iqr_val = q3 - q1

    return ESRes(
        measure="iqr",
        estimate=float(iqr_val),
        n=len(x),
        extra={"q1": q1, "q3": q3, "median": float(np.median(x))},
    )


iqr_ = iqr_statistic


def cheatsheet() -> str:
    return "iqr_statistic({}) -> Interquartile range statistic."
