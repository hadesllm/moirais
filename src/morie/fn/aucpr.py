# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""AUC-PR (area under precision-recall curve)."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes

_QUOTE = "Study the past if you would define the future. -- Confucius"


def auc_pr(y_true, y_score, **kwargs) -> ESRes:
    """
    Compute Area Under the Precision-Recall Curve via trapezoidal rule.

    :param y_true: array-like of true binary labels (0/1).
    :param y_score: array-like of predicted scores/probabilities.
    :return: ESRes with AUC-PR value.

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
    precisions = [1.0]
    recalls = [0.0]
    tp, fp = 0, 0
    for label in yt_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        prec = tp / (tp + fp)
        rec = tp / total_pos
        precisions.append(prec)
        recalls.append(rec)
    auc_val = 0.0
    for i in range(1, len(recalls)):
        auc_val += abs(recalls[i] - recalls[i - 1]) * (precisions[i] + precisions[i - 1]) / 2.0
    return ESRes(
        measure="auc_pr",
        estimate=float(auc_val),
        n=len(yt),
        extra={"n_pos": total_pos, "n_neg": int(np.sum(yt == 0))},
    )


aucpr = auc_pr


def cheatsheet() -> str:
    return "auc_pr({}) -> AUC-PR (area under precision-recall curve)."
