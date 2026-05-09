"""SIU director recommendation analysis."""

from __future__ import annotations

from collections import Counter

import numpy as np

from moirais.fn._containers import DescriptiveResult


def siu_recommendation(
    recommendations: list[str] | np.ndarray,
) -> DescriptiveResult:
    """Analyse SIU director recommendations.

    Parameters
    ----------
    recommendations : array-like
        Director recommendation labels.

    Returns
    -------
    DescriptiveResult
    """
    recs = list(recommendations)
    if len(recs) == 0:
        raise ValueError("recommendations must be non-empty")
    counts = dict(Counter(recs))
    total = len(recs)
    return DescriptiveResult(
        name="siu_recommendation",
        value=float(total),
        extra={"counts": counts, "proportions": {k: v / total for k, v in counts.items()}, "n": total},
    )


siurcm = siu_recommendation


def cheatsheet() -> str:
    return "siu_recommendation({}) -> SIU director recommendation analysis."
