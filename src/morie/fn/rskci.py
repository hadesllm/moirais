# morie.fn -- function file (hadesllm/morie)
"""Concordance (C-statistic) for risk scores."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import ESRes


def risk_concordance(
    scores: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
) -> ESRes:
    """Harrell's concordance index for risk scores.

    Parameters
    ----------
    scores : ndarray
        Risk scores (higher = riskier).
    times : ndarray
        Survival times.
    events : ndarray
        Event indicator (1 = event).

    Returns
    -------
    ESRes
        estimate is the C-statistic.
    """
    scores = np.asarray(scores, dtype=float)
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    concordant = 0
    discordant = 0
    tied = 0
    n = len(scores)
    for i in range(n):
        if events[i] == 0:
            continue
        for j in range(n):
            if times[j] <= times[i] and i != j:
                continue
            if scores[i] > scores[j]:
                concordant += 1
            elif scores[i] < scores[j]:
                discordant += 1
            else:
                tied += 1
    total = concordant + discordant + tied
    c = (concordant + 0.5 * tied) / total if total > 0 else 0.5
    return ESRes(
        measure="risk_concordance",
        estimate=float(c),
        n=n,
        extra={"concordant": concordant, "discordant": discordant, "tied": tied},
    )


rskci = risk_concordance


def cheatsheet() -> str:
    return "risk_concordance({}) -> Concordance (C-statistic) for risk scores."
