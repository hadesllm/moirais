"""SIU case trend over years."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from morie.fn._containers import DescriptiveResult


def siu_trend(
    counts: np.ndarray | list[int],
    years: np.ndarray | list[int] | None = None,
) -> DescriptiveResult:
    """Analyse SIU case trend over years.

    Parameters
    ----------
    counts : array-like
        Annual SIU case counts.
    years : array-like, optional

    Returns
    -------
    DescriptiveResult
    """
    y = np.asarray(counts, dtype=float)
    if len(y) < 3:
        raise ValueError("Need at least 3 years")
    x = np.arange(len(y), dtype=float) if years is None else np.asarray(years, dtype=float)
    slope, intercept, r, p, se = sp_stats.linregress(x, y)
    trend = "increasing" if slope > 0 and p < 0.05 else "decreasing" if slope < 0 and p < 0.05 else "stable"
    return DescriptiveResult(
        name="siu_trend",
        value=float(slope),
        extra={"slope": float(slope), "p_value": float(p), "trend": trend, "n_years": len(y)},
    )


siutrn = siu_trend


def cheatsheet() -> str:
    return "siu_trend({}) -> SIU case trend over years."
