# moirais.fn — function file (hadesllm/moirais)
"""Precision (positive predictive value)."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes

_QUOTE = "Precision is key. -- Grand Admiral Thrawn"


def precision(y_true, y_pred, *, pos_label=1, **kwargs) -> ESRes:
    """
    Compute precision (PPV) for binary classification.

    .. math::

        \\text{Precision} = \\frac{TP}{TP + FP}

    :param y_true: array-like of true labels.
    :param y_pred: array-like of predicted labels.
    :param pos_label: Label considered positive. Default 1.
    :return: ESRes with precision.

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
    fp = int(np.sum((yt != pos_label) & (yp == pos_label)))
    fn = int(np.sum((yt == pos_label) & (yp != pos_label)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return ESRes(
        measure="precision",
        estimate=prec,
        n=len(yt),
        extra={"tp": tp, "fp": fp, "fn": fn, "pos_label": pos_label},
    )


precn = precision


def cheatsheet() -> str:
    return "precision({}) -> Precision (PPV)."
