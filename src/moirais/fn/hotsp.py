# moirais.fn — function file (hadesllm/moirais)
"""Repeat-address / hot-spot analysis. 'It's a trap!' -- Admiral Ackbar"""

from __future__ import annotations

from collections import Counter

import numpy as np

from ._containers import DescriptiveResult


def hot_spots(
    locations: np.ndarray,
    threshold: int = 3,
) -> DescriptiveResult:
    """Identify hot-spot locations that meet a repeat-event threshold.

    Parameters
    ----------
    locations : array-like
        Location identifiers (ints, strings, etc.).
    threshold : int, default 3
        Minimum event count to flag as hot spot.

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``hot_spots`` (array of flagged locations) and
        ``counts`` (full count dict).
    """
    locations = np.asarray(locations)
    counts = Counter(locations.tolist())
    flagged = np.array([loc for loc, cnt in counts.items() if cnt >= threshold])
    return DescriptiveResult(
        name="Hot-spot analysis",
        value=len(flagged),
        extra={
            "hot_spots": flagged,
            "counts": dict(counts),
            "threshold": threshold,
            "total_events": len(locations),
            "n_locations": len(counts),
        },
    )


hotsp = hot_spots


def cheatsheet() -> str:
    return "hot_spots({}) -> Repeat-address / hot-spot analysis. 'It's a trap!' -- Admira"
