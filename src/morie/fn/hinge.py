# morie.fn -- function file (hadesllm/morie)
r"""Hinge loss for SVM.

Encourages correct classification with margin.

References
----------
Crammer, K., & Singer, Y. (2001).
On the algorithmic implementation of multiclass kernel-based vector machines.
JMLR, 2(Dec), 265-292.
"""

__all__ = ["hinge"]

import numpy as np


def hinge(
    y_true,
    y_pred,
    margin=1.0,
):
    """
    Hinge loss.

    Parameters
    ----------
    y_true : ndarray
        True labels, shape (n_samples,). Must be in {-1, 1}.
    y_pred : ndarray
        Decision function output, shape (n_samples,).
    margin : float, optional
        Margin parameter. Default 1.0.

    Returns
    -------
    float
        Mean hinge loss.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have same shape")

    if not np.all(np.isin(y_true, [-1, 1])):
        raise ValueError("y_true must contain only -1 and 1")

    if margin <= 0:
        raise ValueError("margin must be positive")

    loss = np.maximum(margin - y_true * y_pred, 0)

    return float(np.mean(loss))
