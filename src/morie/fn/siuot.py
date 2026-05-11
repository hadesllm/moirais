"""SIU case outcome distribution."""

from __future__ import annotations

from collections import Counter

import numpy as np

from morie.fn._containers import DescriptiveResult


def siu_outcome(
    outcomes: list[str] | np.ndarray,
) -> DescriptiveResult:
    """Analyse SIU case outcome distribution.

    Parameters
    ----------
    outcomes : array-like
        Outcome labels (e.g. 'No charges', 'Charges laid', 'Withdrawn').

    Returns
    -------
    DescriptiveResult
    """
    out = list(outcomes)
    if len(out) == 0:
        raise ValueError("outcomes must be non-empty")
    counts = dict(Counter(out))
    total = len(out)
    props = {k: v / total for k, v in counts.items()}
    return DescriptiveResult(
        name="siu_outcome",
        value=float(total),
        extra={"counts": counts, "proportions": props, "n": total},
    )


siuot = siu_outcome


def cheatsheet() -> str:
    return "siu_outcome({}) -> SIU case outcome distribution."
