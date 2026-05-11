"""Specificity (true negative rate)."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes

_QUOTE = "Numbers have life; they're not just symbols on paper. — Shakuntala Devi"


def specificity(y_true, y_pred, *, pos_label=1, **kwargs) -> ESRes:
    """
    Compute specificity (TNR) for binary classification.

    .. math::

        \\text{Specificity} = \\frac{TN}{TN + FP}

    :param y_true: array-like of true labels.
    :param y_pred: array-like of predicted labels.
    :param pos_label: Label considered positive. Default 1.
    :return: ESRes with specificity.

    References
    ----------
    Altman DG, Bland JM (1994). Diagnostic tests 1: sensitivity and
        specificity. *BMJ*, 308(6943), 1552.
    """
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred must have same length.")
    if len(yt) == 0:
        raise ValueError("Inputs must not be empty.")
    tn = int(np.sum((yt != pos_label) & (yp != pos_label)))
    fp = int(np.sum((yt != pos_label) & (yp == pos_label)))
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return ESRes(
        measure="specificity",
        estimate=spec,
        n=len(yt),
        extra={"tn": tn, "fp": fp, "pos_label": pos_label},
    )


specf = specificity


def cheatsheet() -> str:
    return "specificity({}) -> Specificity (TNR)."
