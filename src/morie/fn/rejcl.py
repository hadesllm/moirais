# morie.fn — function file (hadesllm/morie)
"""Classifier with reject (ambiguity rejection) option."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Judge me by my size, do you?"


def reject_option(scores, threshold=0.5, reject_width=0.1, **kwargs) -> DescriptiveResult:
    """Classify with a reject option for ambiguous samples.

    Samples whose maximum score falls within a rejection band around
    the threshold are marked as rejected (label = -1).

    Parameters
    ----------
    scores : array-like of shape (n,) or (n, k)
        Prediction scores. If 1-D, treated as binary positive class score.
        If 2-D, per-class scores.
    threshold : float
        Decision boundary (default 0.5).
    reject_width : float
        Half-width of the rejection band (default 0.1).

    Returns
    -------
    DescriptiveResult
    """
    scores = np.asarray(scores, dtype=float)

    if scores.ndim == 1:
        max_score = scores
        predictions = np.where(scores >= threshold, 1, 0)
        confidence = np.abs(scores - threshold)
    else:
        max_score = scores.max(axis=1)
        second_max = np.sort(scores, axis=1)[:, -2]
        predictions = scores.argmax(axis=1)
        confidence = max_score - second_max

    rejected = confidence < reject_width
    predictions_with_reject = predictions.copy().astype(float)
    predictions_with_reject[rejected] = -1

    n_rejected = int(rejected.sum())
    reject_rate = float(n_rejected / len(scores)) if len(scores) > 0 else 0.0

    return DescriptiveResult(
        name="reject_option",
        value=reject_rate,
        extra={
            "predictions": predictions_with_reject,
            "rejected_mask": rejected,
            "n_rejected": n_rejected,
            "reject_rate": reject_rate,
            "confidence": confidence,
            "threshold": threshold,
            "reject_width": reject_width,
        },
    )


rejcl = reject_option


def cheatsheet() -> str:
    return "reject_option({}) -> Classifier with reject (ambiguity rejection) option."
