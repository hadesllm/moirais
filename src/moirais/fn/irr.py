# moirais.fn — function file (hadesllm/moirais)
"""Incidence rate ratio (IRR) effect size."""

import math

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def rate_ratio(
    events1: int,
    person_time1: float,
    events2: int,
    person_time2: float,
    confidence: float = 0.95,
) -> ESRes:
    """Incidence rate ratio.

    IRR = (e1/PT1) / (e2/PT2)

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
    irr = r1 / r2 if r2 > 0 else np.inf
    log_irr = math.log(irr) if irr > 0 and np.isfinite(irr) else 0.0
    se = math.sqrt(1 / max(events1, 1) + 1 / max(events2, 1))
    z = stats.norm.ppf((1 + confidence) / 2)
    return ESRes(
        measure="Rate ratio",
        estimate=float(irr),
        ci_lower=float(math.exp(log_irr - z * se)),
        ci_upper=float(math.exp(log_irr + z * se)),
        se=float(se),
        n=events1 + events2,
    )


irr = rate_ratio


def cheatsheet() -> str:
    return "rate_ratio({}) -> Incidence rate ratio (IRR) effect size."
