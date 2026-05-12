# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""AUC-ROC via trapezoidal integration."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes

_QUOTE = "The whole is greater than the sum of its parts. -- Aristotle"


def auc_roc(y_true, y_score, **kwargs) -> ESRes:
    r"""
    Compute Area Under the ROC Curve via trapezoidal rule.

    .. math::

        \\text{AUC} = \\int_0^1 \\text{TPR}(t)\\, d\\text{FPR}(t)

    :param y_true: array-like of true binary labels (0/1).
    :param y_score: array-like of predicted scores/probabilities.
    :return: ESRes with AUC value.

    References
    ----------
    Hanley JA, McNeil BJ (1982). The meaning and use of the area under a
        receiver operating characteristic (ROC) curve. *Radiology*, 143, 29-36.
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    ys = np.asarray(y_score, dtype=float).ravel()
    if len(yt) != len(ys):
        raise ValueError("y_true and y_score must have same length.")
    if len(yt) == 0:
        raise ValueError("Inputs must not be empty.")
    total_pos = np.sum(yt == 1)
    total_neg = np.sum(yt == 0)
    if total_pos == 0 or total_neg == 0:
        raise ValueError("y_true must contain both positive and negative samples.")
    desc_order = np.argsort(-ys)
    yt_sorted = yt[desc_order]
    tp, fp = 0, 0
    fpr_prev, tpr_prev = 0.0, 0.0
    auc = 0.0
    prev_score = None
    for label, score in zip(yt_sorted, ys[desc_order]):
        if score != prev_score and prev_score is not None:
            fpr_cur = fp / total_neg
            tpr_cur = tp / total_pos
            auc += (fpr_cur - fpr_prev) * (tpr_cur + tpr_prev) / 2.0
            fpr_prev, tpr_prev = fpr_cur, tpr_cur
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score
    fpr_cur = fp / total_neg
    tpr_cur = tp / total_pos
    auc += (fpr_cur - fpr_prev) * (tpr_cur + tpr_prev) / 2.0
    return ESRes(
        measure="auc_roc",
        estimate=float(auc),
        n=len(yt),
        extra={"n_pos": int(total_pos), "n_neg": int(total_neg)},
    )


aucroc = auc_roc


def cheatsheet() -> str:
    return "auc_roc({}) -> AUC-ROC via trapezoidal integration."
