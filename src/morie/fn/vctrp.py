"""Repeat victimization analysis."""

from __future__ import annotations

from collections import Counter

import numpy as np

from morie.fn._containers import DescriptiveResult


def victim_repeat(
    victim_ids: list | np.ndarray,
    incident_counts: np.ndarray | list[int] | None = None,
) -> DescriptiveResult:
    """Analyse repeat victimization patterns.

    Parameters
    ----------
    victim_ids : array-like
        Victim identifiers (may repeat).
    incident_counts : array-like, optional
        If provided, number of incidents per victim. Otherwise computed from ids.

    Returns
    -------
    DescriptiveResult
    """
    if incident_counts is not None:
        counts = np.asarray(incident_counts, dtype=int)
    else:
        counter = Counter(victim_ids)
        counts = np.array(list(counter.values()))
    if len(counts) == 0:
        raise ValueError("No victims")
    n_victims = len(counts)
    n_repeat = int(np.sum(counts > 1))
    pct_repeat = n_repeat / n_victims
    return DescriptiveResult(
        name="repeat_victimization",
        value=float(pct_repeat),
        extra={
            "n_victims": n_victims,
            "n_repeat": n_repeat,
            "pct_repeat": pct_repeat,
            "mean_incidents": float(np.mean(counts)),
            "max_incidents": int(np.max(counts)),
        },
    )


vctrp = victim_repeat


def cheatsheet() -> str:
    return "victim_repeat({}) -> Repeat victimization analysis."
