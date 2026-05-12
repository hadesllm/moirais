# morie.fn -- function file (hadesllm/morie)
"""Mental health prevalence trend."""

import numpy as np

from ._containers import DescriptiveResult


def mental_health_trend(
    rates: list | np.ndarray,
    periods: list | np.ndarray,
) -> DescriptiveResult:
    """Analyse trend in mental health prevalence over time.

    Parameters
    ----------
    rates : array-like
        Prevalence rates per period.
    periods : array-like
        Time period labels (numeric).

    Returns
    -------
    DescriptiveResult
    """
    r = np.asarray(rates, dtype=float)
    t = np.asarray(periods, dtype=float)
    if len(r) < 2 or len(r) != len(t):
        raise ValueError("Need >= 2 matched time points")

    slope, intercept = np.polyfit(t, r, 1)
    pct_change = (r[-1] - r[0]) / r[0] * 100 if r[0] != 0 else float("inf")

    return DescriptiveResult(
        name="mental_health_trend",
        value=float(slope),
        extra={"intercept": float(intercept), "pct_change": float(pct_change), "n_periods": len(r)},
    )


mhtrn = mental_health_trend


def cheatsheet() -> str:
    return "mental_health_trend({}) -> Mental health prevalence trend."
