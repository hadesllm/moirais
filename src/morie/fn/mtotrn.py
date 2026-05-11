# morie.fn — function file (hadesllm/morie)
"""Long-term road safety trend."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from morie.fn._containers import DescriptiveResult


def mto_trend(
    yearly_counts: np.ndarray | list[float],
    years: np.ndarray | list[int] | None = None,
) -> DescriptiveResult:
    """Analyse long-term road safety trend.

    Parameters
    ----------
    yearly_counts : array-like
    years : array-like, optional

    Returns
    -------
    DescriptiveResult
    """
    y = np.asarray(yearly_counts, dtype=float)
    if len(y) < 3:
        raise ValueError("Need at least 3 years")
    x = np.arange(len(y), dtype=float) if years is None else np.asarray(years, dtype=float)
    slope, intercept, r, p, se = sp_stats.linregress(x, y)
    trend = "improving" if slope < 0 and p < 0.05 else "worsening" if slope > 0 and p < 0.05 else "stable"
    return DescriptiveResult(
        name="road_safety_trend",
        value=float(slope),
        extra={"slope": float(slope), "p_value": float(p), "r_squared": float(r**2), "trend": trend, "n_years": len(y)},
    )


mtotrn = mto_trend


def cheatsheet() -> str:
    return "mto_trend({}) -> Long-term road safety trend."
