"""Harrell's concordance index for survival models."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def survival_concordance(
    predicted_risk: np.ndarray | list,
    times: np.ndarray | list,
    events: np.ndarray | list,
) -> ESRes:
    """
    Compute Harrell's concordance index (C-index).

    Parameters
    ----------
    predicted_risk : array-like
        Predicted risk scores (higher = worse prognosis).
    times : array-like
        Observed event/censoring times.
    events : array-like
        Event indicators (1 = event).

    Returns
    -------
    ESRes
        estimate = C-index.

    References
    ----------
    Harrell, F. E., et al. (1982). Evaluating the yield of medical
    tests. *JAMA*, 247(18), 2543-2546.
    """
    risk = np.asarray(predicted_risk, dtype=float)
    t = np.asarray(times, dtype=float)
    d = np.asarray(events, dtype=int)
    n = len(t)
    if not (len(risk) == len(t) == len(d)):
        raise ValueError("All arrays must have same length.")

    concordant = 0
    discordant = 0
    tied = 0

    for i in range(n):
        if d[i] == 0:
            continue
        for j in range(n):
            if i == j:
                continue
            if t[j] <= t[i] and d[j] == 0:
                continue
            if t[j] > t[i]:
                if risk[i] > risk[j]:
                    concordant += 1
                elif risk[i] < risk[j]:
                    discordant += 1
                else:
                    tied += 1

    total = concordant + discordant + tied
    c_index = (concordant + 0.5 * tied) / total if total > 0 else 0.5

    return ESRes(
        measure="concordance",
        estimate=float(c_index),
        n=n,
        extra={
            "concordant": concordant,
            "discordant": discordant,
            "tied": tied,
        },
    )


survc = survival_concordance


def cheatsheet() -> str:
    return "survival_concordance({}) -> Harrell's concordance index for survival models."
