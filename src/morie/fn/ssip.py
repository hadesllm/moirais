"""Sample size for comparing two proportions."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import ESRes


def sample_size_proportions(
    p1: float,
    p2: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    ratio: float = 1.0,
) -> ESRes:
    """
    Sample size for a two-sample test comparing proportions.

    Uses the arcsine formula (corrected continuity).

    Parameters
    ----------
    p1 : float
        Expected proportion in group 1.
    p2 : float
        Expected proportion in group 2.
    alpha : float
        Two-sided significance level.
    power : float
        Desired power.
    ratio : float
        Allocation ratio n2/n1.

    Returns
    -------
    ESRes
        estimate = n per group (group 1).

    References
    ----------
    Fleiss, J. L., Levin, B., & Paik, M. C. (2003). *Statistical
    Methods for Rates and Proportions*, 3rd ed. Wiley, Ch. 4.
    """
    if not (0 < p1 < 1) or not (0 < p2 < 1):
        raise ValueError("Proportions must be in (0, 1).")
    if p1 == p2:
        raise ValueError("p1 and p2 must differ.")

    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    p_bar = (p1 + ratio * p2) / (1 + ratio)
    n1 = (
        z_a * np.sqrt(p_bar * (1 - p_bar) * (1 + 1 / ratio)) + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2) / ratio)
    ) ** 2 / (p1 - p2) ** 2
    n1 = int(np.ceil(n1))
    n2 = int(np.ceil(n1 * ratio))

    return ESRes(
        measure="sample_size_proportions",
        estimate=float(n1),
        extra={"n1": n1, "n2": n2, "total": n1 + n2, "p1": p1, "p2": p2},
    )


ssip = sample_size_proportions


def cheatsheet() -> str:
    return "sample_size_proportions({}) -> Sample size for comparing two proportions."
