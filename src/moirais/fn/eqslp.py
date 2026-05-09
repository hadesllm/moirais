# moirais.fn — function file (hadesllm/moirais)
"""Slope index of inequality (SII)."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from moirais.fn._containers import ESRes


def slope_inequality(
    rates: np.ndarray | list[float],
    group_ranks: np.ndarray | list[float],
) -> ESRes:
    """Compute Slope Index of Inequality (SII).

    Linear regression of rate on group rank (0 = most disadvantaged, 1 = most advantaged).

    Parameters
    ----------
    rates : array-like
        Group-specific rates.
    group_ranks : array-like
        Relative group rank (0 to 1 scale).

    Returns
    -------
    ESRes
    """
    r = np.asarray(rates, dtype=float)
    x = np.asarray(group_ranks, dtype=float)
    if len(r) != len(x) or len(r) < 3:
        raise ValueError("Need at least 3 groups, same length")
    slope, intercept, rval, p, se = sp_stats.linregress(x, r)
    return ESRes(
        measure="slope_inequality_index",
        estimate=float(slope),
        se=float(se),
        n=len(r),
        extra={"intercept": float(intercept), "p_value": float(p), "r_squared": float(rval**2)},
    )


eqslp = slope_inequality


def cheatsheet() -> str:
    return "slope_inequality({}) -> Slope index of inequality (SII)."
