# morie.fn -- function file (hadesllm/morie)
"""Geometric mean probability (GMP) fit diagnostic."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def geometric_mean_probability(
    observed_votes,
    predicted_probs,
) -> DescriptiveResult:
    """Geometric mean probability (GMP) for spatial model fit.

    GMP = exp(mean(log(p_correct))) where p_correct is the predicted
    probability of each legislator's actual vote. A perfect model
    achieves GMP = 1; random guessing yields ~0.5.

    :param observed_votes: (n_legislators x n_votes) binary vote matrix.
    :param predicted_probs: (n_legislators x n_votes) predicted P(yea).
    :return: DescriptiveResult with GMP and per-legislator GMPs.

    References
    ----------
    Armstrong (2014), Ch 10. Poole & Rosenthal (1997).

    .. epigraph:: "The greatest teacher, failure is.", Star Wars
    """
    V = np.asarray(observed_votes, dtype=float)
    P = np.asarray(predicted_probs, dtype=float)
    if V.shape != P.shape:
        raise ValueError("observed_votes and predicted_probs must have same shape.")
    if V.ndim != 2:
        raise ValueError("Inputs must be 2D.")

    P = np.clip(P, 1e-10, 1 - 1e-10)
    p_correct = np.where(V == 1, P, 1 - P)
    p_correct = np.where(np.isnan(V), np.nan, p_correct)

    log_p = np.log(p_correct)
    overall_gmp = float(np.exp(np.nanmean(log_p)))

    per_leg = []
    for i in range(V.shape[0]):
        valid = ~np.isnan(log_p[i])
        if valid.sum() > 0:
            per_leg.append(float(np.exp(np.mean(log_p[i, valid]))))
        else:
            per_leg.append(float("nan"))

    per_vote = []
    for j in range(V.shape[1]):
        valid = ~np.isnan(log_p[:, j])
        if valid.sum() > 0:
            per_vote.append(float(np.exp(np.mean(log_p[valid, j]))))
        else:
            per_vote.append(float("nan"))

    return DescriptiveResult(
        name="geometric_mean_probability",
        value=overall_gmp,
        extra={
            "per_legislator_gmp": per_leg,
            "per_vote_gmp": per_vote,
            "n_legislators": V.shape[0],
            "n_votes": V.shape[1],
        },
    )


gmpre = geometric_mean_probability


def cheatsheet() -> str:
    return "geometric_mean_probability({}) -> GMP fit diagnostic for spatial models."
