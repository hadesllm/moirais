# morie.fn — function file (hadesllm/morie)
"""Standardized mortality ratio (SMR) with CI."""

import scipy.stats as stats

from ._containers import ESRes


def standardized_mortality_ratio(
    observed: int,
    expected: float,
    confidence: float = 0.95,
) -> ESRes:
    """Standardized mortality ratio with exact Poisson CI.

    .. math::

        SMR = \\frac{O}{E}

    Parameters
    ----------
    observed : int
    expected : float
    confidence : float

    Returns
    -------
    ESRes
    """
    if expected <= 0:
        raise ValueError("expected must be positive")
    if observed < 0:
        raise ValueError("observed must be non-negative")

    smr = observed / expected
    alpha = 1 - confidence
    ci_lo = stats.chi2.ppf(alpha / 2, 2 * observed) / (2 * expected) if observed > 0 else 0.0
    ci_hi = stats.chi2.ppf(1 - alpha / 2, 2 * (observed + 1)) / (2 * expected)

    return ESRes(
        measure="SMR",
        estimate=float(smr),
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        n=observed,
        extra={"observed": observed, "expected": float(expected)},
    )


cdsmr = standardized_mortality_ratio


def cheatsheet() -> str:
    return "standardized_mortality_ratio({}) -> Standardized mortality ratio (SMR) with CI."
