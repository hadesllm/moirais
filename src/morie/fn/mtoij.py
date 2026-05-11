# morie.fn — function file (hadesllm/morie)
"""Injury severity distribution."""

from __future__ import annotations

from collections import Counter

import numpy as np

from morie.fn._containers import DescriptiveResult


def mto_injury_severity(
    injuries: list[str] | np.ndarray,
) -> DescriptiveResult:
    """Analyse injury severity distribution (fatal/serious/minor/none).

    Parameters
    ----------
    injuries : array-like
        Severity labels per crash.

    Returns
    -------
    DescriptiveResult
    """
    inj = list(injuries)
    if len(inj) == 0:
        raise ValueError("injuries must be non-empty")
    counts = dict(Counter(inj))
    total = len(inj)
    props = {k: v / total for k, v in counts.items()}
    return DescriptiveResult(
        name="injury_severity",
        value=float(total),
        extra={"counts": counts, "proportions": props, "n": total},
    )


mtoij = mto_injury_severity


def cheatsheet() -> str:
    return "mto_injury_severity({}) -> Injury severity distribution."
