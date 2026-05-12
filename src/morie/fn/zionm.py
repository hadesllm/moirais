"""Real knowledge is to know the extent of one's ignorance. -- Confucius"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def defense_allocation(
    threats: np.ndarray | list,
    resources: float = 100.0,
    *,
    risk_aversion: float = 1.0,
) -> DescriptiveResult:
    """Allocate defense resources across threats proportional to risk.

    Uses a risk-weighted proportional allocation model. Each threat has
    a severity score; resources are distributed proportionally to
    severity^risk_aversion.

    Parameters
    ----------
    threats : array-like
        Threat severity scores (positive values).
    resources : float
        Total resources to allocate.
    risk_aversion : float
        Exponent controlling concentration (1 = proportional, >1 = more
        to top threats, <1 = more uniform).

    Returns
    -------
    DescriptiveResult
        ``value`` is the allocation vector (list); ``extra`` has the
        Herfindahl-Hirschman index of concentration.
    """
    t = np.asarray(threats, dtype=np.float64)
    if t.ndim != 1 or len(t) < 1:
        raise ValueError("threats must be a non-empty 1-D array")
    if np.any(t <= 0):
        raise ValueError("All threat scores must be positive")
    if resources <= 0:
        raise ValueError("resources must be positive")

    weights = t**risk_aversion
    weights /= weights.sum()
    allocation = (weights * resources).tolist()

    hhi = float(np.sum(weights**2))
    gini_num = float(np.sum(np.abs(weights[:, None] - weights[None, :])))
    gini = gini_num / (2 * len(t) * weights.sum()) if weights.sum() > 0 else 0

    return DescriptiveResult(
        name="Defense Allocation",
        value=allocation,
        extra={
            "weights": weights.tolist(),
            "hhi": hhi,
            "gini": float(gini),
            "n_threats": len(t),
            "total_resources": resources,
            "risk_aversion": risk_aversion,
        },
    )


zionm = defense_allocation


def cheatsheet() -> str:
    return "Real knowledge is to know the extent of one's ignorance. -- Confucius"
