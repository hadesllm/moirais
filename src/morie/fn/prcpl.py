# morie.fn -- function file (hadesllm/morie)
"""Precision-recall curve."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Who's the more foolish, the fool or the fool who follows him?"


def precision_recall_curve(y_true, y_scores, **kwargs) -> DescriptiveResult:
    """Compute precision-recall curve and average precision.

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

    desc = np.argsort(-y_scores)
    y_sorted = y_true[desc]

    P = np.sum(y_true == 1)
    if P == 0:
        raise ValueError("No positive samples.")

    precisions = [1.0]
    recalls = [0.0]
    tp = 0
    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        prec = tp / (i + 1)
        rec = tp / P
        precisions.append(prec)
        recalls.append(rec)

    precisions = np.array(precisions)
    recalls = np.array(recalls)

    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]

    return DescriptiveResult(
        name="precision_recall_curve",
        value=float(ap),
        extra={
            "precision": precisions,
            "recall": recalls,
            "average_precision": float(ap),
        },
    )


prcpl = precision_recall_curve


def cheatsheet() -> str:
    return "precision_recall_curve({}) -> Precision-recall curve."
