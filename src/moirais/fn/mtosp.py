# moirais.fn — function file (hadesllm/moirais)
"""Speed distribution analysis."""

from __future__ import annotations

import numpy as np

from moirais.fn._containers import DescriptiveResult


def mto_speed_analysis(
    speeds: np.ndarray | list[float],
    *,
    speed_limit: float | None = None,
) -> DescriptiveResult:
    """Analyse speed distribution in crashes or speed surveys.

    Parameters
    ----------
    speeds : array-like
        Speed values (km/h).
    speed_limit : float, optional
        Posted speed limit for over-limit proportion.

    Returns
    -------
    DescriptiveResult
    """
    s = np.asarray(speeds, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) == 0:
        raise ValueError("No valid speed data")
    extra = {
        "mean": float(np.mean(s)),
        "median": float(np.median(s)),
        "std": float(np.std(s, ddof=1)) if len(s) > 1 else 0.0,
        "p85": float(np.percentile(s, 85)),
        "n": len(s),
    }
    if speed_limit is not None:
        extra["pct_over_limit"] = float(np.mean(s > speed_limit))
        extra["speed_limit"] = speed_limit
    return DescriptiveResult(name="speed_analysis", value=float(np.mean(s)), extra=extra)


mtosp = mto_speed_analysis


def cheatsheet() -> str:
    return "mto_speed_analysis({}) -> Speed distribution analysis."
