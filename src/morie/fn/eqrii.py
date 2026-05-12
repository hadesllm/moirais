# morie.fn -- function file (hadesllm/morie)
"""Relative index of inequality (RII)."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from morie.fn._containers import ESRes


def relative_inequality(
    rates: np.ndarray | list[float],
    group_ranks: np.ndarray | list[float],
) -> ESRes:
    """Compute Relative Index of Inequality (RII).

    RII = SII / mean_rate. Values > 1 indicate the gradient
    spans more than the average rate.

    Parameters
    ----------
    rates : array-like
    group_ranks : array-like

    Returns
    -------
    ESRes
    """
    r = np.asarray(rates, dtype=float)
    x = np.asarray(group_ranks, dtype=float)
    if len(r) != len(x) or len(r) < 3:
        raise ValueError("Need at least 3 groups, same length")
    slope, intercept, rval, p, se = sp_stats.linregress(x, r)
    mean_rate = float(np.mean(r))
    rii = slope / mean_rate if mean_rate != 0 else float("inf")
    return ESRes(
        measure="relative_inequality_index",
        estimate=float(rii),
        n=len(r),
        extra={"sii": float(slope), "mean_rate": mean_rate, "p_value": float(p)},
    )


eqrii = relative_inequality


def cheatsheet() -> str:
    return "relative_inequality({}) -> Relative index of inequality (RII)."
