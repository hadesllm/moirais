"""Case clearance rate by offense type."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from morie.fn._containers import CrimeResult


def tps_clearance_rate(
    n_cleared: int,
    n_reported: int,
    *,
    confidence: float = 0.95,
) -> CrimeResult:
    """Compute case clearance rate with Wilson CI.

    Parameters
    ----------
    n_cleared : int
        Number of cases cleared.
    n_reported : int
        Number of cases reported.
    confidence : float
        Confidence level for interval.

    Returns
    -------
    CrimeResult
    """
    if n_reported <= 0:
        raise ValueError("n_reported must be positive")
    if n_cleared < 0 or n_cleared > n_reported:
        raise ValueError("n_cleared must be in [0, n_reported]")
    p = n_cleared / n_reported
    z = sp_stats.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + z**2 / n_reported
    centre = (p + z**2 / (2 * n_reported)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n_reported)) / n_reported) / denom
    return CrimeResult(
        name="clearance_rate",
        rate=p,
        ci_lower=float(max(0, centre - margin)),
        ci_upper=float(min(1, centre + margin)),
        n=n_cleared,
        population=n_reported,
    )


tpscl = tps_clearance_rate


def cheatsheet() -> str:
    return "tps_clearance_rate({}) -> Case clearance rate by offense type."
