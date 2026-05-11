# morie.fn — function file (hadesllm/morie)
"""Precision, recall, and F1 score."""

import numpy as np

from ._containers import ESRes

_QUOTE = "Without music, life would be a mistake. — Friedrich Nietzsche"


def precision_recall(y_true, y_pred, **kwargs) -> ESRes:
    """
    Compute precision, recall, and F1 score for binary classification.

    .. math::

        \\text{Precision} = \\frac{TP}{TP + FP}, \\quad
        \\text{Recall} = \\frac{TP}{TP + FN}, \\quad
        F_1 = \\frac{2 \\cdot P \\cdot R}{P + R}

    :param y_true: array-like of true binary labels.
    :param y_pred: array-like of predicted binary labels.
    :return: ESRes with F1 as estimate, precision/recall in extra.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length.")
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return ESRes(
        measure="precision_recall",
        estimate=f1,
        n=len(y_true),
        extra={"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn},
    )


prrcl = precision_recall


def cheatsheet() -> str:
    return "precision_recall({}) -> Precision, recall, and F1 score."
