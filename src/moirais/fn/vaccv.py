"""Vaccination coverage rate."""

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def vaccine_coverage(
    n_vaccinated: int,
    n_eligible: int,
    confidence: float = 0.95,
) -> ESRes:
    """Compute vaccination coverage with Wilson CI.

    Parameters
    ----------
    n_vaccinated : int
    n_eligible : int
    confidence : float

    Returns
    -------
    ESRes
    """
    if n_eligible <= 0:
        raise ValueError("n_eligible must be positive")
    if n_vaccinated < 0 or n_vaccinated > n_eligible:
        raise ValueError("n_vaccinated must be in [0, n_eligible]")

    p = n_vaccinated / n_eligible
    z = stats.norm.ppf((1 + confidence) / 2)
    denom = 1 + z**2 / n_eligible
    centre = p + z**2 / (2 * n_eligible)
    margin = z * np.sqrt(p * (1 - p) / n_eligible + z**2 / (4 * n_eligible**2))

    return ESRes(
        measure="vaccine_coverage",
        estimate=float(p),
        ci_lower=float((centre - margin) / denom),
        ci_upper=float((centre + margin) / denom),
        n=n_eligible,
    )


vaccv = vaccine_coverage


def cheatsheet() -> str:
    return "vaccine_coverage({}) -> Vaccination coverage rate."
