"""Youden's J index for optimal classification threshold."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def youden_index(
    y_true: np.ndarray | list,
    y_scores: np.ndarray | list,
) -> ESRes:
    """
    Find optimal threshold via Youden's J statistic.

    .. math::

        J = \\max_c \\{\\text{Sens}(c) + \\text{Spec}(c) - 1\\}

    Parameters
    ----------
    y_true : array-like
        Binary labels (0/1).
    y_scores : array-like
        Predicted scores.

    Returns
    -------
    ESRes
        estimate = J, extra has 'optimal_threshold', 'sensitivity',
        'specificity'.

    References
    ----------
    Youden, W. J. (1950). Index for rating diagnostic tests.
    *Cancer*, 3(1), 32-35.
    """
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_scores, dtype=float)
    if len(y) != len(scores):
        raise ValueError("Arrays must have same length.")

    thresholds = np.unique(scores)
    best_j = -1.0
    best_thresh = 0.0
    best_sens = 0.0
    best_spec = 0.0

    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = np.sum((pred == 1) & (y == 1))
        tn = np.sum((pred == 0) & (y == 0))
        fp = np.sum((pred == 1) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = sens + spec - 1
        if j > best_j:
            best_j = j
            best_thresh = t
            best_sens = sens
            best_spec = spec

    return ESRes(
        measure="youden_J",
        estimate=float(best_j),
        n=len(y),
        extra={
            "optimal_threshold": float(best_thresh),
            "sensitivity": float(best_sens),
            "specificity": float(best_spec),
        },
    )


youdn = youden_index


def cheatsheet() -> str:
    return "youden_index({}) -> Youden's J index for optimal classification threshold."
