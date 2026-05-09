# moirais.fn — function file (hadesllm/moirais)
"""Incidence rate difference (IRD) effect size."""

import math

import scipy.stats as stats

from ._containers import ESRes


def incidence_rate_difference(
    events1: int,
    person_time1: float,
    events2: int,
    person_time2: float,
    confidence: float = 0.95,
) -> ESRes:
    """Incidence rate difference.

    IRD = (e1/PT1) - (e2/PT2)

    Parameters
    ----------
    events1, person_time1 : int, float
    events2, person_time2 : int, float
    confidence : float, default 0.95

    Returns
    -------
    ESRes
    """
    r1 = events1 / person_time1 if person_time1 > 0 else 0.0
    r2 = events2 / person_time2 if person_time2 > 0 else 0.0
    ird = r1 - r2
    se = (
        math.sqrt(events1 / person_time1**2 + events2 / person_time2**2)
        if person_time1 > 0 and person_time2 > 0
        else 0.0
    )
    z = stats.norm.ppf((1 + confidence) / 2)
    return ESRes(
        measure="Incidence rate difference",
        estimate=float(ird),
        ci_lower=float(ird - z * se),
        ci_upper=float(ird + z * se),
        se=float(se),
    )


ird = incidence_rate_difference


def cheatsheet() -> str:
    return "incidence_rate_difference({}) -> Incidence rate difference (IRD) effect size."
