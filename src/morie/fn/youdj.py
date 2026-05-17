"""Youden's J statistic (informedness)."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes

_QUOTE = "The whole is greater than the sum of its parts. -- Aristotle"


def youdens_j(y_true, y_pred, *, pos_label=1, **kwargs) -> ESRes:
    """
    Compute Youden's J statistic (informedness).

    .. math::

        J = \\text{Sensitivity} + \\text{Specificity} - 1

    Range [-1, 1]. J = 0 means useless test, J = 1 means perfect.

    :param y_true: array-like of true labels.
    :param y_pred: array-like of predicted labels.
    :param pos_label: Label considered positive. Default 1.
    :return: ESRes with J statistic.

    References
    ----------
    Youden WJ (1950). Index for rating diagnostic tests. *Cancer*, 3(1), 32-35.
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
    j = sens + spec - 1.0
    return ESRes(
        measure="youdens_j",
        estimate=j,
        n=len(yt),
        extra={"sensitivity": sens, "specificity": spec},
    )


youdj = youdens_j


def cheatsheet() -> str:
    return "youdens_j({}) -> Youden's J statistic."
