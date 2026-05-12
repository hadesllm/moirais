# morie.fn -- function file (hadesllm/morie)
"""Concordance index (C-statistic) for survival models."""

from __future__ import annotations

import numpy as np

__all__ = ["clndn"]


def clndn(
    predicted_risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
) -> dict:
    """Harrell's concordance index (C-index).

    Parameters
    ----------
    predicted_risk : array-like
        Predicted risk scores (higher = worse prognosis).
    time : array-like
        Observed event/censoring times.
    event : array-like
        Event indicator (1=event).

    Returns
    -------
    dict
        c_index, concordant, discordant, tied, se, n_pairs.
    """
    risk = np.asarray(predicted_risk, dtype=float)
    t = np.asarray(time, dtype=float)
    d = np.asarray(event, dtype=int)
    n = len(t)

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
    se = np.sqrt(c_index * (1 - c_index) / max(total, 1))

    return {
        "c_index": float(c_index),
        "concordant": concordant,
        "discordant": discordant,
        "tied": tied,
        "se": float(se),
        "n_pairs": total,
    }


clndn_fn = clndn


def cheatsheet() -> str:
    return "clndn(predicted_risk, time, event) -> Concordance index (C-statistic)."
