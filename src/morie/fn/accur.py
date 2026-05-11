# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Accuracy from confusion matrix."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes
def accuracy(y_true, y_pred, **kwargs) -> ESRes:
    """
    Compute classification accuracy.

    .. math::

        \\text{Accuracy} = \\frac{TP + TN}{TP + TN + FP + FN}

    :param y_true: array-like of true labels (0/1).
    :param y_pred: array-like of predicted labels (0/1).
    :return: ESRes with accuracy.

    References
    ----------
    Sokolova M, Lapalme G (2009). A systematic analysis of performance
        measures for classification tasks. *Information Processing & Management*.
    """
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred must have same length.")
    n = len(yt)
    if n == 0:
        raise ValueError("Inputs must not be empty.")
    correct = int(np.sum(yt == yp))
    acc = correct / n
    tp = int(np.sum((yt == 1) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    return ESRes(
        measure="accuracy",
        estimate=acc,
        n=n,
        extra={"tp": tp, "tn": tn, "fp": fp, "fn": fn, "correct": correct},
    )


accur = accuracy


def cheatsheet() -> str:
    return "accuracy({}) -> Classification accuracy."
