# morie.fn — function file (hadesllm/morie)
"""Maternal mortality ratio."""

from __future__ import annotations

import scipy.stats as stats

from ._containers import ESRes


def maternal_mortality_ratio(
    maternal_deaths: int,
    live_births: int,
    confidence: float = 0.95,
    per: int = 100000,
) -> ESRes:
    """Maternal mortality ratio (MMR).

    .. math::

        MMR = \\frac{\\text{maternal deaths}}{\\text{live births}} \\times 100{,}000

    Parameters
    ----------
    maternal_deaths : int
        Deaths from maternal causes.
    live_births : int
        Total live births.
    confidence : float, default 0.95
        Confidence level.
    per : int, default 100000
        Rate multiplier.

    Returns
    -------
    ESRes

    References
    ----------
    WHO (2019). Trends in maternal mortality 2000 to 2017.
    World Health Organization.
    """
    if live_births <= 0:
        raise ValueError("live_births must be positive")
    if maternal_deaths < 0:
        raise ValueError("maternal_deaths must be non-negative")

    ratio = maternal_deaths / live_births * per
    alpha = 1 - confidence

    if maternal_deaths == 0:
        ci_lo = 0.0
        ci_hi = stats.chi2.ppf(1 - alpha / 2, 2) / (2 * live_births) * per
    else:
        ci_lo = stats.chi2.ppf(alpha / 2, 2 * maternal_deaths) / (2 * live_births) * per
        ci_hi = stats.chi2.ppf(1 - alpha / 2, 2 * (maternal_deaths + 1)) / (2 * live_births) * per

    return ESRes(
        measure="MMR",
        estimate=float(ratio),
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        n=live_births,
        extra={"maternal_deaths": maternal_deaths, "per": per},
    )


mmrat = maternal_mortality_ratio


def cheatsheet() -> str:
    return "maternal_mortality_ratio({}) -> Maternal mortality ratio with CI."
