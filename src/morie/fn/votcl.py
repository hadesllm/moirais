"""Majority / weighted voting classifier."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Let the Wookiee win."


def voting_classify(predictions, weights=None, **kwargs) -> DescriptiveResult:
    """Majority or weighted voting ensemble.

    Parameters
    ----------
    predictions : array-like of shape (n_classifiers, n_samples)
        Each row is one classifier's predictions.
    weights : array-like of shape (n_classifiers,) or None
        Voting weights. Default: uniform.

    Returns
    -------
    DescriptiveResult
    """
    preds = np.asarray(predictions)
    if preds.ndim == 1:
        preds = preds.reshape(1, -1)
    n_clf, n_samples = preds.shape

    if weights is None:
        weights = np.ones(n_clf)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    classes = np.unique(preds)
    final = np.empty(n_samples, dtype=preds.dtype)

    for j in range(n_samples):
        votes = {}
        for i in range(n_clf):
            c = preds[i, j]
            votes[c] = votes.get(c, 0.0) + weights[i]
        final[j] = max(votes, key=votes.get)

    agreement = float(np.mean([np.sum(preds[:, j] == final[j]) / n_clf for j in range(n_samples)]))

    return DescriptiveResult(
        name="voting_classify",
        value=agreement,
        extra={
            "predictions": final,
            "agreement_rate": agreement,
            "n_classifiers": n_clf,
            "weights": weights,
        },
    )


votcl = voting_classify


def cheatsheet() -> str:
    return "voting_classify({}) -> Majority / weighted voting classifier."
