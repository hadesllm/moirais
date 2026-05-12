# morie.fn -- function file (hadesllm/morie)
"""Quality-adjusted life years (QALY)."""

import numpy as np

from ._containers import ESRes


def quality_adjusted_ly(
    utilities: list | np.ndarray,
    durations: list | np.ndarray,
) -> ESRes:
    r"""Compute QALYs from utility weights and durations.

    .. math::

        QALY = \\sum_i u_i \\times d_i

    Parameters
    ----------
    utilities : array-like
        Health state utility values (0-1).
    durations : array-like
        Time spent in each health state (years).

    Returns
    -------
    ESRes
    """
    u = np.asarray(utilities, dtype=float)
    d = np.asarray(durations, dtype=float)
    if len(u) != len(d):
        raise ValueError("utilities and durations must match")
    if np.any(u < 0) or np.any(u > 1):
        raise ValueError("utilities must be in [0, 1]")

    qaly = float(np.sum(u * d))
    total_time = float(np.sum(d))

    return ESRes(
        measure="QALY",
        estimate=qaly,
        extra={"total_time": total_time, "mean_utility": float(np.average(u, weights=d)) if total_time > 0 else 0.0},
    )


heqly = quality_adjusted_ly


def cheatsheet() -> str:
    return "quality_adjusted_ly({}) -> Quality-adjusted life years (QALY)."
