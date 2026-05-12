# morie.fn -- function file (hadesllm/morie)
"""Recall (sensitivity / true positive rate)."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes

_QUOTE = "Your eyes can deceive you. Don't trust them. --"


def recall(y_true, y_pred, *, pos_label=1, **kwargs) -> ESRes:
    """
    Compute recall (sensitivity / TPR) for binary classification.

    .. math::

        \\text{Recall} = \\frac{TP}{TP + FN}

    :param y_true: array-like of true labels.
    :param y_pred: array-like of predicted labels.
    :param pos_label: Label considered positive. Default 1.
    :return: ESRes with recall.

    References
    ----------
    Powers DMW (2011). Evaluation: from precision, recall and F-measure
        to ROC, informedness, markedness and correlation. *JMLT*, 2(1).
    """
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred must have same length.")
    if len(yt) == 0:
        raise ValueError("Inputs must not be empty.")
    tp = int(np.sum((yt == pos_label) & (yp == pos_label)))
    fn = int(np.sum((yt == pos_label) & (yp != pos_label)))
    fp = int(np.sum((yt != pos_label) & (yp == pos_label)))
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return ESRes(
        measure="recall",
        estimate=rec,
        n=len(yt),
        extra={"tp": tp, "fn": fn, "fp": fp, "pos_label": pos_label},
    )


recal = recall


def cheatsheet() -> str:
    return "recall({}) -> Recall (sensitivity / TPR)."
