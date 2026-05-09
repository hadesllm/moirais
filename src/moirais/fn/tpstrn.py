"""Crime trend analysis."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from moirais.fn._containers import DescriptiveResult


def tps_crime_trend(
    counts: np.ndarray | list[float],
    periods: np.ndarray | list | None = None,
) -> DescriptiveResult:
    """Analyse temporal crime trend via linear regression.

    Parameters
    ----------
    counts : array-like
        Crime counts per period.
    periods : array-like, optional
        Period labels/indices. Defaults to 0..n-1.

    Returns
    -------
    DescriptiveResult
        With extra keys slope, p_value, trend.
    """
    y = np.asarray(counts, dtype=float)
    if len(y) < 3:
        raise ValueError("Need at least 3 periods for trend")
    x = np.arange(len(y), dtype=float) if periods is None else np.asarray(periods, dtype=float)
    slope, intercept, r, p, se = sp_stats.linregress(x, y)
    trend = "increasing" if slope > 0 and p < 0.05 else "decreasing" if slope < 0 and p < 0.05 else "stable"
    return DescriptiveResult(
        name="crime_trend",
        value=float(slope),
        extra={
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r**2),
            "p_value": float(p),
            "se": float(se),
            "trend": trend,
            "n_periods": len(y),
        },
    )


tpstrn = tps_crime_trend


def cheatsheet() -> str:
    return "tps_crime_trend({}) -> Crime trend analysis."
