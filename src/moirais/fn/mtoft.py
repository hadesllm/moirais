# moirais.fn — function file (hadesllm/moirais)
"""Traffic fatality rate per 100K population."""

from __future__ import annotations

from scipy import stats as sp_stats

from moirais.fn._containers import CrimeResult


def mto_fatality_rate(
    n_fatalities: int,
    population: int,
    *,
    per: int = 100000,
    confidence: float = 0.95,
) -> CrimeResult:
    """Compute traffic fatality rate per 100K with Poisson CI.

    Parameters
    ----------
    n_fatalities : int
    population : int
    per : int
    confidence : float

    Returns
    -------
    CrimeResult
    """
    if population <= 0:
        raise ValueError("population must be positive")
    rate = n_fatalities / population * per
    alpha = 1 - confidence
    ci_lo = sp_stats.chi2.ppf(alpha / 2, 2 * n_fatalities) / 2 / population * per if n_fatalities > 0 else 0.0
    ci_hi = sp_stats.chi2.ppf(1 - alpha / 2, 2 * (n_fatalities + 1)) / 2 / population * per
    return CrimeResult(
        name="fatality_rate", rate=rate, ci_lower=ci_lo, ci_upper=ci_hi, n=n_fatalities, population=population
    )


mtoft = mto_fatality_rate


def cheatsheet() -> str:
    return "mto_fatality_rate({}) -> Traffic fatality rate per 100K population."
