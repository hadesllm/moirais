"""AUDIT alcohol screening score."""

import numpy as np

from ._containers import ESRes


def audit_score(
    items: list | np.ndarray,
) -> ESRes:
    """Compute AUDIT (Alcohol Use Disorders Identification Test) score.

    The AUDIT has 10 items scored 0-4, yielding total 0-40.
    Risk zones: 0-7 low, 8-15 hazardous, 16-19 harmful, 20-40 dependence.

    Parameters
    ----------
    items : array-like
        Ten item responses, each in [0, 4].

    Returns
    -------
    ESRes

    References
    ----------
    Saunders, J. B. et al. (1993). Development of the AUDIT.
    Addiction, 88(6), 791-804.
    """
    a = np.asarray(items, dtype=int)
    if len(a) != 10:
        raise ValueError("AUDIT requires exactly 10 items")
    if np.any(a < 0) or np.any(a > 4):
        raise ValueError("Each item must be in [0, 4]")

    total = int(np.sum(a))
    if total <= 7:
        zone = "low_risk"
    elif total <= 15:
        zone = "hazardous"
    elif total <= 19:
        zone = "harmful"
    else:
        zone = "probable_dependence"

    return ESRes(
        measure="AUDIT",
        estimate=float(total),
        extra={
            "zone": zone,
            "consumption": int(np.sum(a[:3])),
            "dependence": int(np.sum(a[3:6])),
            "harm": int(np.sum(a[6:])),
        },
    )


suaud = audit_score


def cheatsheet() -> str:
    return "audit_score({}) -> AUDIT alcohol screening score."
