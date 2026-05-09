# moirais.fn — function file (hadesllm/moirais)
"""Roll call analysis for legislative voting."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def roll_call_analysis(vote_matrix) -> DescriptiveResult:
    """Analyse a legislative roll-call vote matrix.

    Computes per-legislator and per-vote summary statistics:
    loyalty, participation, marginals, lopsidedness.

    :param vote_matrix: (n_legislators x n_votes) binary matrix (1=yea, 0=nay, NaN=absent).
    :return: DescriptiveResult with summary statistics.

    References
    ----------
    Armstrong (2014), Ch 7.

    .. epigraph:: "So this is how liberty dies, with thunderous applause." -- Padme, Star Wars
    """
    V = np.asarray(vote_matrix, dtype=float)
    if V.ndim != 2:
        raise ValueError("vote_matrix must be 2D.")
    n_leg, n_votes = V.shape

    participation = np.array([np.sum(~np.isnan(V[i])) / n_votes for i in range(n_leg)])
    yea_rates = np.array([np.nanmean(V[i]) for i in range(n_leg)])

    vote_margins = []
    lopsided = []
    for j in range(n_votes):
        valid = V[~np.isnan(V[:, j]), j]
        if len(valid) > 0:
            yea_pct = valid.mean()
            vote_margins.append(float(yea_pct))
            lopsided.append(abs(yea_pct - 0.5) > 0.35)
        else:
            vote_margins.append(float("nan"))
            lopsided.append(False)

    n_close = sum(1 for m in vote_margins if not np.isnan(m) and abs(m - 0.5) < 0.1)

    agree = np.zeros((n_leg, n_leg))
    for i in range(n_leg):
        for j in range(i + 1, n_leg):
            mask = ~np.isnan(V[i]) & ~np.isnan(V[j])
            if mask.sum() > 0:
                agree[i, j] = agree[j, i] = float(np.mean(V[i, mask] == V[j, mask]))

    return DescriptiveResult(
        name="roll_call_analysis",
        value={
            "participation_rates": participation,
            "yea_rates": yea_rates,
            "vote_margins": vote_margins,
        },
        extra={
            "n_legislators": n_leg,
            "n_votes": n_votes,
            "n_close_votes": n_close,
            "n_lopsided": sum(lopsided),
            "mean_agreement": float(agree[np.triu_indices(n_leg, k=1)].mean()) if n_leg > 1 else 0.0,
        },
    )


rollc = roll_call_analysis


def cheatsheet() -> str:
    return "roll_call_analysis({}) -> Roll call analysis for legislative voting."
