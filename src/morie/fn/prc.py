# morie.fn — function file (hadesllm/morie)
"""Precision-recall curve."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def pr_curve(
    y_true: np.ndarray | list,
    y_scores: np.ndarray | list,
) -> DescriptiveResult:
    """
    Compute precision-recall curve from binary labels and scores.

    Parameters
    ----------
    y_true : array-like
        Binary labels (0/1).
    y_scores : array-like
        Predicted scores or probabilities.

    Returns
    -------
    DescriptiveResult
        extra has 'precision', 'recall', 'thresholds', 'auprc'.

    References
    ----------
    Davis, J., & Goadrich, M. (2006). The relationship between
    Precision-Recall and ROC curves. *ICML*, 233-240.
    """
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_scores, dtype=float)
    if len(y) != len(scores):
        raise ValueError("y_true and y_scores must have same length.")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("y_true must be binary (0/1).")

    desc_idx = np.argsort(-scores)
    y_sorted = y[desc_idx]
    scores_sorted = scores[desc_idx]

    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)
    n_pos = y.sum()

    precision = tp_cum / (tp_cum + fp_cum)
    recall = tp_cum / n_pos if n_pos > 0 else tp_cum

    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auprc = float(_trapz(precision, recall))

    return DescriptiveResult(
        name="PR_curve",
        value=abs(auprc),
        extra={
            "precision": precision,
            "recall": recall,
            "thresholds": scores_sorted,
            "auprc": abs(auprc),
        },
    )


prc = pr_curve


def cheatsheet() -> str:
    return "pr_curve({}) -> Precision-recall curve."
