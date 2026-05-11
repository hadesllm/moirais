"""DAST drug screening score."""

import numpy as np

from ._containers import ESRes


def dast_score(
    items: list | np.ndarray,
) -> ESRes:
    """Compute DAST-10 (Drug Abuse Screening Test) score.

    10 binary items (0/1). Total 0-10.
    Zones: 0 none, 1-2 low, 3-5 moderate, 6-8 substantial, 9-10 severe.

    Parameters
    ----------
    items : array-like
        Ten binary responses (0 or 1).

    Returns
    -------
    ESRes

    References
    ----------
    Skinner, H. A. (1982). The Drug Abuse Screening Test.
    Addictive Behaviors, 7(4), 363-371.
    """
    a = np.asarray(items, dtype=int)
    if len(a) != 10:
        raise ValueError("DAST-10 requires exactly 10 items")
    if not np.all((a == 0) | (a == 1)):
        raise ValueError("Each item must be 0 or 1")

    total = int(np.sum(a))
    if total == 0:
        zone = "no_problems"
    elif total <= 2:
        zone = "low"
    elif total <= 5:
        zone = "moderate"
    elif total <= 8:
        zone = "substantial"
    else:
        zone = "severe"

    return ESRes(measure="DAST-10", estimate=float(total), extra={"zone": zone})


sudast = dast_score


def cheatsheet() -> str:
    return "dast_score({}) -> DAST drug screening score."
