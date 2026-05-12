# morie.fn -- function file (hadesllm/morie)
"""Cumulative incidence (risk)."""

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def cumulative_incidence(
    n_events: int,
    n_at_risk: int,
    follow_up_time: float = 1.0,
    confidence: float = 0.95,
) -> ESRes:
    """Cumulative incidence proportion with Wilson CI.

    Parameters
    ----------
    n_events : int
    n_at_risk : int
    follow_up_time : float
        Follow-up duration for context.
    confidence : float

    Returns
    -------
    ESRes
    """
    if n_at_risk <= 0:
        raise ValueError("n_at_risk must be positive")
    if n_events < 0 or n_events > n_at_risk:
        raise ValueError("n_events must be in [0, n_at_risk]")

    p = n_events / n_at_risk
    z = stats.norm.ppf((1 + confidence) / 2)
    denom = 1 + z**2 / n_at_risk
    centre = p + z**2 / (2 * n_at_risk)
    margin = z * np.sqrt(p * (1 - p) / n_at_risk + z**2 / (4 * n_at_risk**2))

    return ESRes(
        measure="cumulative_incidence",
        estimate=float(p),
        ci_lower=float((centre - margin) / denom),
        ci_upper=float((centre + margin) / denom),
        n=n_at_risk,
        extra={"follow_up_time": follow_up_time},
    )


cdcum = cumulative_incidence


def cheatsheet() -> str:
    return "cumulative_incidence({}) -> Cumulative incidence (risk)."
