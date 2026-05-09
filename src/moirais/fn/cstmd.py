# moirais.fn — function file (hadesllm/moirais)
"""Medical service utilization rate in custody."""

from __future__ import annotations

import numpy as np

from moirais.fn._containers import CrimeResult


def custody_medical(
    visits: np.ndarray,
    person_days: np.ndarray,
    *,
    rate_per: int = 1000,
) -> CrimeResult:
    """Medical service utilization rate per 1000 person-days.

    Parameters
    ----------
    visits : ndarray
        Visit counts per unit.
    person_days : ndarray
        Person-days per unit.
    rate_per : int
        Rate denominator (default 1000).

    Returns
    -------
    CrimeResult
    """
    visits = np.asarray(visits, dtype=float)
    person_days = np.asarray(person_days, dtype=float)
    total_visits = float(np.sum(visits))
    total_pd = float(np.sum(person_days))
    rate = total_visits / max(total_pd, 1e-10) * rate_per
    se = np.sqrt(total_visits) / max(total_pd, 1e-10) * rate_per
    return CrimeResult(
        name="custody_medical",
        rate=rate,
        n=int(total_visits),
        population=int(total_pd),
        ci_lower=max(rate - 1.96 * se, 0),
        ci_upper=rate + 1.96 * se,
    )


cstmd = custody_medical


def cheatsheet() -> str:
    return "custody_medical({}) -> Medical service utilization rate in custody."
