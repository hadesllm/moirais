# morie.fn -- function file (hadesllm/morie)
"""Classification rate for spatial voting model."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def classification_rate(
    observed_votes,
    ideal_points,
    alternatives,
    *,
    model: str = "quadratic",
    beta: float = 1.0,
) -> DescriptiveResult:
    """Classification rate: proportion of votes correctly predicted.

    Compares predicted votes from a spatial model against observed votes
    to assess model fit.

    :param observed_votes: (n_voters x n_votes) binary vote matrix (1=yea).
    :param ideal_points: (n_voters x n_dims) estimated ideal points.
    :param alternatives: (n_votes x 2 x n_dims) yea/nay midpoints per vote.
    :param model: Utility model ('quadratic' or 'gaussian').
    :param beta: Spatial weight parameter.
    :return: DescriptiveResult with overall and per-vote classification rates.

    References
    ----------
    Armstrong (2014), Ch 10.

    .. epigraph:: Give me a place to stand and I will move the earth. -- Archimedes
    """
    V = np.asarray(observed_votes, dtype=float)
    X = np.asarray(ideal_points, dtype=float)
    A = np.asarray(alternatives, dtype=float)

    if V.ndim != 2:
        raise ValueError("observed_votes must be 2D.")
    n_voters, n_votes = V.shape

    correct = 0
    total = 0
    per_vote_rates = []
    for j in range(n_votes):
        yea_pos = A[j, 0]
        nay_pos = A[j, 1]
        d_yea = np.sum((X - yea_pos) ** 2, axis=1)
        d_nay = np.sum((X - nay_pos) ** 2, axis=1)
        if model == "gaussian":
            pred = (np.exp(-0.5 * beta * d_yea) > np.exp(-0.5 * beta * d_nay)).astype(float)
        else:
            pred = (d_yea < d_nay).astype(float)

        mask = ~np.isnan(V[:, j])
        n_valid = mask.sum()
        n_correct = int((pred[mask] == V[mask, j]).sum())
        correct += n_correct
        total += n_valid
        per_vote_rates.append(n_correct / max(n_valid, 1))

    overall = correct / max(total, 1)
    return DescriptiveResult(
        name="classification_rate",
        value=overall,
        extra={
            "per_vote_rates": per_vote_rates,
            "n_correct": correct,
            "n_total": total,
        },
    )


clfrt = classification_rate


def cheatsheet() -> str:
    return "classification_rate({}) -> Classification rate for spatial voting model."
