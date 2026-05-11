"""Substance use trend over time periods."""

import numpy as np

from ._containers import DescriptiveResult


def substance_trend(
    rates: list | np.ndarray,
    periods: list | np.ndarray,
) -> DescriptiveResult:
    """Analyse substance use trends over time periods.

    Fits a linear trend via OLS and returns the slope, intercept,
    and percent change from first to last period.

    Parameters
    ----------
    rates : array-like
        Substance use rates per period.
    periods : array-like
        Time period labels (numeric, e.g. years).

    Returns
    -------
    DescriptiveResult
    """
    r = np.asarray(rates, dtype=float)
    t = np.asarray(periods, dtype=float)
    if len(r) < 2:
        raise ValueError("Need at least 2 time points")
    if len(r) != len(t):
        raise ValueError("rates and periods must have same length")

    slope, intercept = np.polyfit(t, r, 1)
    pct_change = (r[-1] - r[0]) / r[0] * 100 if r[0] != 0 else float("inf")

    return DescriptiveResult(
        name="substance_trend",
        value=float(slope),
        extra={
            "intercept": float(intercept),
            "pct_change": float(pct_change),
            "n_periods": len(r),
            "rates": r.tolist(),
            "periods": t.tolist(),
        },
    )


sutrn = substance_trend


def cheatsheet() -> str:
    return "substance_trend({}) -> Substance use trend over time periods."
