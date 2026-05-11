# morie.fn — function file (hadesllm/morie)
"""Time to trial distribution."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import DescriptiveResult


def court_time_to_trial(
    days_to_trial: np.ndarray | list[float],
) -> DescriptiveResult:
    """Analyse time-to-trial distribution.

    Parameters
    ----------
    days_to_trial : array-like
        Days from charge to trial for each case.

    Returns
    -------
    DescriptiveResult
    """
    d = np.asarray(days_to_trial, dtype=float)
    d = d[np.isfinite(d) & (d >= 0)]
    if len(d) == 0:
        raise ValueError("No valid durations")
    return DescriptiveResult(
        name="time_to_trial",
        value=float(np.median(d)),
        extra={
            "mean_days": float(np.mean(d)),
            "median_days": float(np.median(d)),
            "std_days": float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
            "p75_days": float(np.percentile(d, 75)),
            "p90_days": float(np.percentile(d, 90)),
            "pct_over_365": float(np.mean(d > 365)),
            "n": len(d),
        },
    )


crttm = court_time_to_trial


def cheatsheet() -> str:
    return "court_time_to_trial({}) -> Time to trial distribution."
