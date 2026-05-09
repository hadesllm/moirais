# moirais.fn — function file (hadesllm/moirais)
"""Guilty plea rate."""

from __future__ import annotations

from collections import Counter

import numpy as np

from moirais.fn._containers import CrimeResult


def court_plea_rate(
    plea_types: list[str] | np.ndarray,
) -> CrimeResult:
    """Compute guilty plea rate.

    Parameters
    ----------
    plea_types : array-like
        Plea labels (e.g. 'Guilty', 'Not guilty', 'No contest').

    Returns
    -------
    CrimeResult
    """
    pleas = list(plea_types)
    if len(pleas) == 0:
        raise ValueError("plea_types must be non-empty")
    counts = Counter(p.lower() for p in pleas)
    n_guilty = sum(v for k, v in counts.items() if "guilty" in k and "not" not in k)
    total = len(pleas)
    rate = n_guilty / total
    return CrimeResult(
        name="guilty_plea_rate",
        rate=rate,
        n=n_guilty,
        population=total,
        extra={"plea_counts": dict(Counter(pleas))},
    )


crtpl = court_plea_rate


def cheatsheet() -> str:
    return "court_plea_rate({}) -> Guilty plea rate."
