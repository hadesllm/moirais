# morie.fn — function file (hadesllm/morie)
"""Precision-recall curve."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The truth is often what we make of it. --"


def pr_curve(y_true, y_score, **kwargs) -> DescriptiveResult:
    """
    Compute precision-recall curve at varying thresholds.

    :param y_true: array-like of true binary labels (0/1).
    :param y_score: array-like of predicted scores/probabilities.
    :return: DescriptiveResult with precision, recall, and thresholds.

    References
    ----------
    Davis J, Goadrich M (2006). The relationship between precision-recall
        and ROC curves. *Proc. ICML*.
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    ys = np.asarray(y_score, dtype=float).ravel()
    if len(yt) != len(ys):
        raise ValueError("y_true and y_score must have same length.")
    if len(yt) == 0:
        raise ValueError("Inputs must not be empty.")
    total_pos = int(np.sum(yt == 1))
    if total_pos == 0:
        raise ValueError("y_true must contain at least one positive sample.")
    desc_order = np.argsort(-ys)
    yt_sorted = yt[desc_order]
    ys_sorted = ys[desc_order]
    precisions = [1.0]
    recalls = [0.0]
    thresholds = []
    tp, fp = 0, 0
    prev_score = None
    for label, score in zip(yt_sorted, ys_sorted):
        if score != prev_score and prev_score is not None:
            prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            rec = tp / total_pos
            precisions.append(prec)
            recalls.append(rec)
            thresholds.append(float(score))
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / total_pos
    precisions.append(prec)
    recalls.append(rec)
    thresholds.append(float(ys_sorted[-1]))
    return DescriptiveResult(
        name="pr_curve",
        value=None,
        extra={
            "precision": precisions,
            "recall": recalls,
            "thresholds": thresholds,
            "n": len(yt),
        },
    )


prcv = pr_curve


def cheatsheet() -> str:
    return "pr_curve({}) -> Precision-recall curve."
