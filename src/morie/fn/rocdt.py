# morie.fn — function file (hadesllm/morie)
"""ROC and DET curves with AUC."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "I've got a bad feeling about this."


def roc_det_curve(y_true, y_scores, **kwargs) -> DescriptiveResult:
    """Compute ROC and DET curves with AUC.

    ROC plots TPR vs FPR. DET plots FNR vs FPR (both on normal deviate scale).

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary ground truth (0/1).
    y_scores : array-like of shape (n,)
        Predicted scores (higher = more positive).

    Returns
    -------
    DescriptiveResult
    """
    y_true = np.asarray(y_true).ravel()
    y_scores = np.asarray(y_scores, dtype=float).ravel()

    desc = np.argsort(-y_scores)
    y_sorted = y_true[desc]
    scores_sorted = y_scores[desc]

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    if P == 0 or N == 0:
        raise ValueError("Need both positive and negative samples.")

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp = 0
    fp = 0
    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / P)
        fpr_list.append(fp / N)

    fpr = np.array(fpr_list)
    tpr = np.array(tpr_list)
    fnr = 1.0 - tpr

    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = float(_trapz(tpr, fpr))

    return DescriptiveResult(
        name="roc_det_curve",
        value=auc,
        extra={
            "fpr": fpr,
            "tpr": tpr,
            "fnr": fnr,
            "auc": auc,
            "thresholds": scores_sorted,
        },
    )


rocdt = roc_det_curve


def cheatsheet() -> str:
    return "roc_det_curve({}) -> ROC and DET curves with AUC."
