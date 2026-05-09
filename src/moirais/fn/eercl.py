# moirais.fn — function file (hadesllm/moirais)
"""Equal error rate (EER) computation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "So this is how liberty dies, with thunderous applause."


def equal_error_rate(y_true, y_scores, **kwargs) -> DescriptiveResult:
    """Compute the equal error rate where FAR equals FRR.

    EER is the operating point on the ROC where false acceptance rate
    equals false rejection rate.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary labels (0/1).
    y_scores : array-like of shape (n,)
        Predicted scores.

    Returns
    -------
    DescriptiveResult
    """
    y_true = np.asarray(y_true).ravel()
    y_scores = np.asarray(y_scores, dtype=float).ravel()

    thresholds = np.sort(np.unique(y_scores))
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    if P == 0 or N == 0:
        raise ValueError("Need both classes present.")

    best_diff = np.inf
    eer = 0.0
    eer_thresh = 0.0

    for t in thresholds:
        far = np.sum((y_scores >= t) & (y_true == 0)) / N
        frr = np.sum((y_scores < t) & (y_true == 1)) / P
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            eer = (far + frr) / 2.0
            eer_thresh = float(t)

    return DescriptiveResult(
        name="equal_error_rate",
        value=float(eer),
        extra={
            "eer": float(eer),
            "threshold": eer_thresh,
            "n_positive": int(P),
            "n_negative": int(N),
        },
    )


eercl = equal_error_rate


def cheatsheet() -> str:
    return "equal_error_rate({}) -> Equal error rate (EER) computation."
