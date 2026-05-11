# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Balanced accuracy."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes
def balanced_accuracy(y_true, y_pred, *, pos_label=1, **kwargs) -> ESRes:
    """
    Compute balanced accuracy (average of sensitivity and specificity).

    .. math::

        \\text{BA} = \\frac{\\text{Sensitivity} + \\text{Specificity}}{2}

    Useful when classes are imbalanced.

    :param y_true: array-like of true labels.
    :param y_pred: array-like of predicted labels.
    :param pos_label: Label considered positive. Default 1.
    :return: ESRes with balanced accuracy.

    References
    ----------
    Brodersen KH et al. (2010). The balanced accuracy and its posterior
        distribution. *ICPR*.
    """
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred must have same length.")
    if len(yt) == 0:
        raise ValueError("Inputs must not be empty.")
    tp = int(np.sum((yt == pos_label) & (yp == pos_label)))
    fn = int(np.sum((yt == pos_label) & (yp != pos_label)))
    tn = int(np.sum((yt != pos_label) & (yp != pos_label)))
    fp = int(np.sum((yt != pos_label) & (yp == pos_label)))
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ba = (sens + spec) / 2.0
    return ESRes(
        measure="balanced_accuracy",
        estimate=ba,
        n=len(yt),
        extra={"sensitivity": sens, "specificity": spec, "tp": tp, "tn": tn, "fp": fp, "fn": fn},
    )


balan = balanced_accuracy


def cheatsheet() -> str:
    return "balanced_accuracy({}) -> Balanced accuracy."
