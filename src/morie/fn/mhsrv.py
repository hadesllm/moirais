# morie.fn — function file (hadesllm/morie)
"""Mental health service utilization rate."""

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def service_utilization(
    n_accessing: int,
    n_need: int,
    confidence: float = 0.95,
) -> ESRes:
    """Mental health service utilization rate (treatment gap).

    Parameters
    ----------
    n_accessing : int
        Number accessing services.
    n_need : int
        Number with identified need.
    confidence : float

    Returns
    -------
    ESRes
    """
    if n_need <= 0:
        raise ValueError("n_need must be positive")
    if n_accessing < 0 or n_accessing > n_need:
        raise ValueError("n_accessing must be in [0, n_need]")

    p = n_accessing / n_need
    z = stats.norm.ppf((1 + confidence) / 2)
    se = np.sqrt(p * (1 - p) / n_need)
    gap = 1 - p

    return ESRes(
        measure="service_utilization",
        estimate=float(p),
        ci_lower=float(max(0, p - z * se)),
        ci_upper=float(min(1, p + z * se)),
        se=float(se),
        n=n_need,
        extra={"treatment_gap": float(gap), "n_unmet": n_need - n_accessing},
    )


mhsrv = service_utilization


def cheatsheet() -> str:
    return "service_utilization({}) -> Mental health service utilization rate."
