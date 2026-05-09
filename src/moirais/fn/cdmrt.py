# moirais.fn — function file (hadesllm/moirais)
"""Cause-specific mortality rate."""

import scipy.stats as stats

from ._containers import ESRes


def cause_specific_mortality(
    n_deaths_cause: int,
    population: int,
    per: int = 100000,
    confidence: float = 0.95,
) -> ESRes:
    """Cause-specific mortality rate per *per* population.

    Parameters
    ----------
    n_deaths_cause : int
    population : int
    per : int
        Denominator (default 100,000).
    confidence : float

    Returns
    -------
    ESRes
    """
    if population <= 0:
        raise ValueError("population must be positive")

    rate = n_deaths_cause / population * per
    alpha = 1 - confidence
    ci_lo = stats.chi2.ppf(alpha / 2, 2 * n_deaths_cause) / (2 * population) * per if n_deaths_cause > 0 else 0.0
    ci_hi = stats.chi2.ppf(1 - alpha / 2, 2 * (n_deaths_cause + 1)) / (2 * population) * per

    return ESRes(
        measure="cause_specific_mortality_rate",
        estimate=float(rate),
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        n=population,
        extra={"per": per},
    )


cdmrt = cause_specific_mortality


def cheatsheet() -> str:
    return "cause_specific_mortality({}) -> Cause-specific mortality rate."
