# moirais.fn — function file (hadesllm/moirais)
"""Infant mortality rate."""

from __future__ import annotations

import scipy.stats as stats

from ._containers import ESRes


def infant_mortality_rate(
    infant_deaths: int,
    live_births: int,
    confidence: float = 0.95,
    per: int = 1000,
) -> ESRes:
    """Infant mortality rate (deaths < 1 year per live births).

    .. math::

        IMR = \\frac{\\text{deaths}_{<1}}{\\text{live births}} \\times 1000

    Parameters
    ----------
    infant_deaths : int
        Deaths under age 1.
    live_births : int
        Total live births.
    confidence : float, default 0.95
        Confidence level.
    per : int, default 1000
        Rate multiplier.

    Returns
    -------
    ESRes

    References
    ----------
    Reidpath, D. D. & Allotey, P. (2003). Infant mortality rate as
    an indicator of population health. Journal of Epidemiology and
    Community Health, 57(5), 344-346.
    """
    if live_births <= 0:
        raise ValueError("live_births must be positive")
    if infant_deaths < 0 or infant_deaths > live_births:
        raise ValueError("infant_deaths must be in [0, live_births]")

    rate = infant_deaths / live_births * per
    alpha = 1 - confidence

    if infant_deaths == 0:
        ci_lo = 0.0
        ci_hi = stats.chi2.ppf(1 - alpha / 2, 2) / (2 * live_births) * per
    else:
        ci_lo = stats.chi2.ppf(alpha / 2, 2 * infant_deaths) / (2 * live_births) * per
        ci_hi = stats.chi2.ppf(1 - alpha / 2, 2 * (infant_deaths + 1)) / (2 * live_births) * per

    return ESRes(
        measure="IMR",
        estimate=float(rate),
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        n=live_births,
        extra={"infant_deaths": infant_deaths, "per": per},
    )


infrt = infant_mortality_rate


def cheatsheet() -> str:
    return "infant_mortality_rate({}) -> Infant mortality rate with CI."
