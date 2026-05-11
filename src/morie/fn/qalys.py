# morie.fn — function file (hadesllm/morie)
"""Quality-adjusted life years (QALY) computation."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def qaly_computation(
    durations: list[float] | np.ndarray,
    utilities: list[float] | np.ndarray,
    discount_rate: float = 0.03,
) -> ESRes:
    """Compute quality-adjusted life years (QALYs).

    .. math::

        QALY = \\sum_i u_i \\cdot d_i \\cdot (1+r)^{-i}

    Parameters
    ----------
    durations : array-like of float
        Duration (years) of each health state.
    utilities : array-like of float
        Utility weight (0-1) for each health state.
    discount_rate : float, default 0.03
        Annual discount rate.

    Returns
    -------
    ESRes

    References
    ----------
    Weinstein, M. C. et al. (2009). QALYs: the basics. Value in
    Health, 12(s1), S5-S9.
    """
    dur = np.asarray(durations, dtype=float)
    util = np.asarray(utilities, dtype=float)

    if len(dur) != len(util):
        raise ValueError("durations and utilities must match")
    if np.any(util < 0) or np.any(util > 1):
        raise ValueError("Utilities must be in [0, 1]")
    if np.any(dur < 0):
        raise ValueError("Durations must be non-negative")

    r = discount_rate
    total_qaly = 0.0
    undiscounted = 0.0
    cum_time = 0.0

    for i in range(len(dur)):
        undiscounted += dur[i] * util[i]
        if r > 0 and dur[i] > 0:
            disc = (1 - np.exp(-r * dur[i])) / r * np.exp(-r * cum_time)
        else:
            disc = dur[i]
        total_qaly += util[i] * disc
        cum_time += dur[i]

    return ESRes(
        measure="QALY",
        estimate=float(total_qaly),
        extra={
            "undiscounted": float(undiscounted),
            "discount_rate": r,
            "total_duration": float(np.sum(dur)),
            "mean_utility": float(np.mean(util)),
        },
    )


qalys = qaly_computation


def cheatsheet() -> str:
    return "qaly_computation({}) -> Quality-adjusted life years (QALY)."
