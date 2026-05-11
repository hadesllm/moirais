"""Stability check for a discrete-time system."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "I have a bad feeling about this."


def stability_check(b, a) -> DescriptiveResult:
    """Check BIBO stability: all poles must lie inside the unit circle.

    Parameters
    ----------
    b : array-like
        Numerator coefficients.
    a : array-like
        Denominator coefficients.

    Returns
    -------
    DescriptiveResult
        value is 1.0 if stable, 0.0 otherwise.
    """
    a = np.asarray(a, dtype=float)
    poles = np.roots(a)
    max_pole_mag = float(np.max(np.abs(poles))) if len(poles) > 0 else 0.0
    stable = max_pole_mag < 1.0
    return DescriptiveResult(
        name="stability_check",
        value=1.0 if stable else 0.0,
        extra={"stable": stable, "max_pole_magnitude": max_pole_mag, "poles": poles},
    )


stbck = stability_check


def cheatsheet() -> str:
    return "stability_check({}) -> Stability check for a discrete-time system."
