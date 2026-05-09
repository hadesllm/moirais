# moirais.fn — function file (hadesllm/moirais)
"""Mental health disorder prevalence with CI."""

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def mental_health_prevalence(
    n_cases: int,
    n_surveyed: int,
    confidence: float = 0.95,
) -> ESRes:
    """Estimate prevalence of mental health disorders with Wilson CI.

    Parameters
    ----------
    n_cases : int
    n_surveyed : int
    confidence : float

    Returns
    -------
    ESRes
    """
    if n_surveyed <= 0:
        raise ValueError("n_surveyed must be positive")
    if n_cases < 0 or n_cases > n_surveyed:
        raise ValueError("n_cases must be in [0, n_surveyed]")

    p = n_cases / n_surveyed
    z = stats.norm.ppf((1 + confidence) / 2)
    denom = 1 + z**2 / n_surveyed
    centre = p + z**2 / (2 * n_surveyed)
    margin = z * np.sqrt(p * (1 - p) / n_surveyed + z**2 / (4 * n_surveyed**2))

    return ESRes(
        measure="mental_health_prevalence",
        estimate=float(p),
        ci_lower=float((centre - margin) / denom),
        ci_upper=float((centre + margin) / denom),
        n=n_surveyed,
    )


mhprv = mental_health_prevalence


def cheatsheet() -> str:
    return "mental_health_prevalence({}) -> Mental health disorder prevalence with CI."
