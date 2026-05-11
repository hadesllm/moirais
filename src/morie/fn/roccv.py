# morie.fn — function file (hadesllm/morie)
"""ROC curve (FPR, TPR at thresholds)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult
def roc_curve(y_true, y_score, **kwargs) -> DescriptiveResult:
    """
    Compute Receiver Operating Characteristic curve.

    Sweeps thresholds from max to min of y_score, computing FPR and TPR
    at each unique threshold.

    :param y_true: array-like of true binary labels (0/1).
    :param y_score: array-like of predicted scores/probabilities.
    :return: DescriptiveResult with fpr, tpr, and thresholds arrays.

    References
    ----------
    Fawcett T (2006). An introduction to ROC analysis. *Pattern Recognition
        Letters*, 27(8), 861-874.
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    ys = np.asarray(y_score, dtype=float).ravel()
    if len(yt) != len(ys):
        raise ValueError("y_true and y_score must have same length.")
    if len(yt) == 0:
        raise ValueError("Inputs must not be empty.")
    desc_order = np.argsort(-ys)
    yt_sorted = yt[desc_order]
    ys_sorted = ys[desc_order]
    total_pos = np.sum(yt == 1)
    total_neg = np.sum(yt == 0)
    if total_pos == 0 or total_neg == 0:
        raise ValueError("y_true must contain both positive and negative samples.")
    thresholds = []
    fpr_list = []
    tpr_list = []
    fpr_list.append(0.0)
    tpr_list.append(0.0)
    thresholds.append(ys_sorted[0] + 1.0)
    tp, fp = 0, 0
    prev_score = None
    for i, (label, score) in enumerate(zip(yt_sorted, ys_sorted)):
        if score != prev_score and prev_score is not None:
            fpr_list.append(fp / total_neg)
            tpr_list.append(tp / total_pos)
            thresholds.append(score)
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score
    fpr_list.append(fp / total_neg)
    tpr_list.append(tp / total_pos)
    thresholds.append(ys_sorted[-1] - 1.0)
    return DescriptiveResult(
        name="roc_curve",
        value=None,
        extra={
            "fpr": fpr_list,
            "tpr": tpr_list,
            "thresholds": thresholds,
            "n": len(yt),
        },
    )


roccv = roc_curve


def cheatsheet() -> str:
    return "roc_curve({}) -> ROC curve (FPR, TPR at thresholds)."
