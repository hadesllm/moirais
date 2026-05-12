# morie.fn -- function file (hadesllm/morie)
"""Maternal mortality ratio."""

import scipy.stats as stats

from ._containers import ESRes


def maternal_mortality(
    n_maternal_deaths: int,
    n_live_births: int,
    per: int = 100000,
    confidence: float = 0.95,
) -> ESRes:
    """Maternal mortality ratio per *per* live births.

    Parameters
    ----------
    n_maternal_deaths : int
    n_live_births : int
    per : int
    confidence : float

    Returns
    -------
    ESRes
    """
    if n_live_births <= 0:
        raise ValueError("n_live_births must be positive")

    rate = n_maternal_deaths / n_live_births * per
    alpha = 1 - confidence
    ci_lo = (
        stats.chi2.ppf(alpha / 2, 2 * n_maternal_deaths) / (2 * n_live_births) * per if n_maternal_deaths > 0 else 0.0
    )
    ci_hi = stats.chi2.ppf(1 - alpha / 2, 2 * (n_maternal_deaths + 1)) / (2 * n_live_births) * per

    return ESRes(
        measure="maternal_mortality_ratio",
        estimate=float(rate),
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        n=n_live_births,
        extra={"per": per},
    )


mmort = maternal_mortality


def cheatsheet() -> str:
    return "maternal_mortality({}) -> Maternal mortality ratio."
