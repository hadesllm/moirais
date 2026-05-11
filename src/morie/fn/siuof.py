"""He who is brave is free. — Seneca"""

from __future__ import annotations

from collections import Counter

import numpy as np

from morie.fn._containers import DescriptiveResult


def siu_officer_force(
    force_types: list[str] | np.ndarray,
) -> DescriptiveResult:
    """Analyse use of force types in SIU-investigated incidents.

    Parameters
    ----------
    force_types : array-like
        Force type labels (e.g. 'Firearm discharge', 'CEW', 'Physical').

    Returns
    -------
    DescriptiveResult
    """
    ft = list(force_types)
    if len(ft) == 0:
        raise ValueError("force_types must be non-empty")
    counts = dict(Counter(ft))
    total = len(ft)
    props = {k: v / total for k, v in counts.items()}
    return DescriptiveResult(
        name="siu_officer_force",
        value=float(total),
        extra={"counts": counts, "proportions": props, "n": total},
    )


siuof = siu_officer_force


def cheatsheet() -> str:
    return "siu_officer_force({}) -> Use of force in SIU cases analysis."
