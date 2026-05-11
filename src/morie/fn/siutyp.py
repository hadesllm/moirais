"""SIU cases by incident type."""

from __future__ import annotations

from collections import Counter

import numpy as np

from morie.fn._containers import DescriptiveResult


def siu_by_type(
    types: list[str] | np.ndarray,
) -> DescriptiveResult:
    """Analyse SIU cases by incident type.

    Parameters
    ----------
    types : array-like
        Incident type labels (shooting, custody death, sexual assault, vehicle).

    Returns
    -------
    DescriptiveResult
    """
    t = list(types)
    if len(t) == 0:
        raise ValueError("types must be non-empty")
    counts = dict(Counter(t))
    total = len(t)
    return DescriptiveResult(
        name="siu_by_type",
        value=float(total),
        extra={"counts": counts, "proportions": {k: v / total for k, v in counts.items()}, "n": total},
    )


siutyp = siu_by_type


def cheatsheet() -> str:
    return "siu_by_type({}) -> SIU cases by incident type."
