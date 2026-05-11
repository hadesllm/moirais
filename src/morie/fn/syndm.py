"""Syndromic surveillance composite score."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def syndromic_score(
    symptom_counts: dict[str, int | float],
    *,
    weights: dict[str, float] | None = None,
) -> ESRes:
    """
    Compute a composite syndromic surveillance score.

    Weighted sum of symptom category counts, normalised by total.

    Parameters
    ----------
    symptom_counts : dict
        {symptom_category: count}.
    weights : dict, optional
        {symptom_category: weight}. Default: equal weights.

    Returns
    -------
    ESRes
        estimate = composite score.

    References
    ----------
    Henning, K. J. (2004). What is syndromic surveillance?
    *MMWR Suppl*, 53, 5-11.
    """
    if not symptom_counts:
        raise ValueError("symptom_counts must not be empty.")

    cats = list(symptom_counts.keys())
    vals = np.array([float(symptom_counts[c]) for c in cats])

    if weights is None:
        w = np.ones(len(cats))
    else:
        w = np.array([float(weights.get(c, 1.0)) for c in cats])

    total = float(np.sum(vals))
    weighted_sum = float(np.dot(w, vals))
    score = weighted_sum / float(np.sum(w)) if np.sum(w) > 0 else 0.0

    return ESRes(
        measure="syndromic_score",
        estimate=score,
        n=int(total),
        extra={"weighted_sum": weighted_sum, "categories": cats},
    )


syndm = syndromic_score


def cheatsheet() -> str:
    return "syndromic_score({}) -> Syndromic surveillance composite score."
