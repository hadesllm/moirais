"""Substance-attributable mortality."""

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def substance_mortality(
    deaths_attributed: int,
    total_deaths: int,
    confidence: float = 0.95,
) -> ESRes:
    r"""Compute substance-attributable fraction of mortality.

    .. math::

        SAF = \\frac{\\text{deaths\\_attributed}}{\\text{total\\_deaths}}

    Parameters
    ----------
    deaths_attributed : int
        Deaths attributable to substance use.
    total_deaths : int
        Total deaths in the population.
    confidence : float

    Returns
    -------
    ESRes
    """
    if total_deaths <= 0:
        raise ValueError("total_deaths must be positive")
    if deaths_attributed < 0 or deaths_attributed > total_deaths:
        raise ValueError("deaths_attributed must be in [0, total_deaths]")

    p = deaths_attributed / total_deaths
    z = stats.norm.ppf((1 + confidence) / 2)
    se = np.sqrt(p * (1 - p) / total_deaths)

    return ESRes(
        measure="substance_attributable_fraction",
        estimate=float(p),
        ci_lower=float(max(0, p - z * se)),
        ci_upper=float(min(1, p + z * se)),
        se=float(se),
        n=total_deaths,
    )


sumort = substance_mortality


def cheatsheet() -> str:
    return "substance_mortality({}) -> Substance-attributable mortality."
