# morie.fn -- function file (hadesllm/morie)
"""Proportional reduction in error (PRE) for spatial models."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def proportional_reduction_error(
    observed_votes,
    predicted_votes,
) -> DescriptiveResult:
    """Proportional reduction in error (PRE) for spatial voting models.

    PRE = (errors_null - errors_model) / errors_null, where the null
    model predicts the majority outcome for every vote. Also called
    APRE (aggregate PRE).

    :param observed_votes: (n_legislators x n_votes) binary vote matrix.
    :param predicted_votes: (n_legislators x n_votes) model predictions (binary).
    :return: DescriptiveResult with PRE, APRE, and classification rates.

    References
    ----------
    Armstrong (2014), Ch 10.

    .. epigraph:: "Once you start down the dark path, forever will it dominate your destiny."
    """
    V = np.asarray(observed_votes, dtype=float)
    P = np.asarray(predicted_votes, dtype=float)
    if V.shape != P.shape:
        raise ValueError("observed and predicted must have same shape.")
    if V.ndim != 2:
        raise ValueError("Inputs must be 2D.")
    n_leg, n_votes = V.shape

    model_errors = 0
    null_errors = 0
    total = 0
    per_vote_pre = []

    for j in range(n_votes):
        valid = ~np.isnan(V[:, j])
        n_valid = int(valid.sum())
        if n_valid == 0:
            per_vote_pre.append(float("nan"))
            continue
        obs = V[valid, j]
        pred = P[valid, j]
        yea_count = obs.sum()
        nay_count = n_valid - yea_count
        null_pred = 1.0 if yea_count >= nay_count else 0.0
        n_err = int((obs != null_pred).sum())
        m_err = int((obs != pred).sum())
        null_errors += n_err
        model_errors += m_err
        total += n_valid
        if n_err > 0:
            per_vote_pre.append(float((n_err - m_err) / n_err))
        else:
            per_vote_pre.append(1.0)

    pre = float((null_errors - model_errors) / max(null_errors, 1))
    clf_rate = 1.0 - model_errors / max(total, 1)
    null_rate = 1.0 - null_errors / max(total, 1)

    return DescriptiveResult(
        name="proportional_reduction_error",
        value=pre,
        extra={
            "model_errors": model_errors,
            "null_errors": null_errors,
            "classification_rate": clf_rate,
            "null_classification_rate": null_rate,
            "per_vote_pre": per_vote_pre,
        },
    )


prech = proportional_reduction_error


def cheatsheet() -> str:
    return "proportional_reduction_error({}) -> PRE for spatial voting models."
