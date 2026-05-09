"""Standardized mortality/morbidity ratio (SMR)."""

from __future__ import annotations

import scipy.stats as stats

from ._containers import ESRes


def standardized_mortality_ratio(
    observed: int,
    expected: float,
    confidence: float = 0.95,
) -> ESRes:
    """Standardized mortality (or morbidity) ratio.

    SMR = observed / expected, with exact Poisson CI.

    Parameters
    ----------
    observed : int
        Observed number of events.
    expected : float
        Expected number of events (from reference population rates).
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    ESRes

    References
    ----------
    Breslow, N. E. & Day, N. E. (1987). *Statistical Methods in
    Cancer Research*, Vol. II. IARC Scientific Publications No. 82.
    """
    if expected <= 0:
        raise ValueError("expected must be positive")
    if observed < 0:
        raise ValueError("observed must be non-negative")

    smr = observed / expected
    alpha = 1 - confidence

    if observed == 0:
        ci_lo = 0.0
        ci_hi = stats.chi2.ppf(1 - alpha / 2, 2) / (2 * expected)
    else:
        ci_lo = stats.chi2.ppf(alpha / 2, 2 * observed) / (2 * expected)
        ci_hi = stats.chi2.ppf(1 - alpha / 2, 2 * (observed + 1)) / (2 * expected)

    return ESRes(
        measure="SMR",
        estimate=float(smr),
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        n=observed,
        extra={"observed": observed, "expected": float(expected)},
    )


stdmr = standardized_mortality_ratio


def cheatsheet() -> str:
    return "standardized_mortality_ratio({}) -> SMR with exact Poisson CI."
