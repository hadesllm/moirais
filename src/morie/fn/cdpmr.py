# morie.fn — function file (hadesllm/morie)
"""Proportionate mortality ratio."""

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def proportionate_mortality(
    n_deaths_cause: int,
    total_deaths: int,
    confidence: float = 0.95,
) -> ESRes:
    """Proportionate mortality ratio.

    .. math::

        PMR = \\frac{\\text{deaths from cause}}{\\text{total deaths}}

    Parameters
    ----------
    n_deaths_cause : int
    total_deaths : int
    confidence : float

    Returns
    -------
    ESRes
    """
    if total_deaths <= 0:
        raise ValueError("total_deaths must be positive")
    if n_deaths_cause < 0 or n_deaths_cause > total_deaths:
        raise ValueError("n_deaths_cause must be in [0, total_deaths]")

    p = n_deaths_cause / total_deaths
    z = stats.norm.ppf((1 + confidence) / 2)
    se = np.sqrt(p * (1 - p) / total_deaths)

    return ESRes(
        measure="proportionate_mortality_ratio",
        estimate=float(p),
        ci_lower=float(max(0, p - z * se)),
        ci_upper=float(min(1, p + z * se)),
        se=float(se),
        n=total_deaths,
    )


cdpmr = proportionate_mortality


def cheatsheet() -> str:
    return "proportionate_mortality({}) -> Proportionate mortality ratio."
