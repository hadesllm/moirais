"""Excess risk estimation."""

from __future__ import annotations

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def excess_risk(
    observed: int,
    expected: float,
    population: int | None = None,
    confidence: float = 0.95,
    per: int = 100000,
) -> ESRes:
    """Estimate excess risk (observed minus expected events).

    Excess risk quantifies the additional burden beyond what is
    expected based on a reference population or baseline.

    Parameters
    ----------
    observed : int
        Observed number of events.
    expected : float
        Expected number of events.
    population : int, optional
        Population at risk (for rate computation).
    confidence : float, default 0.95
        Confidence level.
    per : int, default 100000
        Rate multiplier.

    Returns
    -------
    ESRes

    References
    ----------
    Rothman, K. J. et al. (2008). *Modern Epidemiology*, 3rd ed.
    Lippincott Williams & Wilkins, Ch. 4.
    """
    if expected < 0:
        raise ValueError("expected must be non-negative")
    if observed < 0:
        raise ValueError("observed must be non-negative")

    excess = observed - expected
    se_excess = np.sqrt(observed) if observed > 0 else 0.0

    z = stats.norm.ppf((1 + confidence) / 2)
    ci_lo = excess - z * se_excess
    ci_hi = excess + z * se_excess

    extra = {
        "observed": observed,
        "expected": float(expected),
        "excess_count": float(excess),
        "SMR": float(observed / expected) if expected > 0 else np.inf,
    }

    estimate = float(excess)
    if population is not None and population > 0:
        excess_rate = excess / population * per
        extra["excess_rate_per"] = float(excess_rate)
        extra["per"] = per
        estimate = float(excess_rate)
        ci_lo = (excess - z * se_excess) / population * per
        ci_hi = (excess + z * se_excess) / population * per

    return ESRes(
        measure="excess_risk",
        estimate=estimate,
        se=float(se_excess),
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        n=observed,
        extra=extra,
    )


xsrsk = excess_risk


def cheatsheet() -> str:
    return "excess_risk({}) -> Excess risk estimation (observed - expected)."
