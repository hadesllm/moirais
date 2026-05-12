# morie.fn -- function file (hadesllm/morie)
"""Indigenous health indicator trend."""

import numpy as np

from ._containers import DescriptiveResult


def indigenous_health_trend(
    rates: list | np.ndarray,
    years: list | np.ndarray,
) -> DescriptiveResult:
    """Analyse trend in Indigenous health indicator over time.

    Parameters
    ----------
    rates : array-like
    years : array-like

    Returns
    -------
    DescriptiveResult
    """
    r = np.asarray(rates, dtype=float)
    y = np.asarray(years, dtype=float)
    if len(r) < 2 or len(r) != len(y):
        raise ValueError("Need >= 2 matched points")

    slope, intercept = np.polyfit(y, r, 1)
    improving = slope < 0 if r[0] > 0 else slope > 0

    return DescriptiveResult(
        name="indigenous_health_trend",
        value=float(slope),
        extra={"intercept": float(intercept), "improving": bool(improving), "n_periods": len(r)},
    )


ihtrn = indigenous_health_trend


def cheatsheet() -> str:
    return "indigenous_health_trend({}) -> Indigenous health indicator trend."
