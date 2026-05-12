# morie.fn -- function file (hadesllm/morie)
"""Crime rate per 100K with Wilson CI. 'These are not the droids you are looking for.'"""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ._containers import CrimeResult


def crime_rate(
    n_crimes: int,
    population: int,
    confidence: float = 0.95,
    per: int = 100_000,
) -> CrimeResult:
    """Crime rate per *per* population with Wilson score interval.

    Parameters
    ----------
    n_crimes : int
        Number of crime events.
    population : int
        Population at risk.
    confidence : float, default 0.95
        Confidence level for the interval.
    per : int, default 100000
        Scaling factor (rate per *per* population).

    Returns
    -------
    CrimeResult
    """
    if population <= 0:
        raise ValueError("population must be positive")
    p_hat = n_crimes / population
    z = _st.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + z**2 / population
    centre = (p_hat + z**2 / (2 * population)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / population + z**2 / (4 * population**2)) / denom
    return CrimeResult(
        name="Crime rate",
        rate=float(p_hat * per),
        ci_lower=float(max(0, (centre - margin) * per)),
        ci_upper=float((centre + margin) * per),
        n=n_crimes,
        population=population,
        extra={"per": per, "confidence": confidence},
    )


crime = crime_rate


def cheatsheet() -> str:
    return "crime_rate({}) -> Crime rate per 100K with Wilson CI. 'These are not the droid"
