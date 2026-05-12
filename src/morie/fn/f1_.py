# morie.fn -- function file (hadesllm/morie)
"""F1 score, precision, and recall."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def f1_score(
    y_true: Union[np.ndarray, Any],
    y_pred: Union[np.ndarray, Any],
) -> dict[str, float]:
    """Compute F1 score, precision, recall, and support.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Ground-truth binary labels (0/1).
    y_pred : array-like of shape (n,)
        Predicted binary labels (0/1).

    Returns
    -------
    dict
        f1, precision, recall, support (number of positive cases).

    References
    ----------
    van Rijsbergen, C. J. (1979). *Information Retrieval* (2nd ed.).
        Butterworths.
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("y_true and y_pred must have same length.")

    tp = float(np.sum((yt == 1) & (yp == 1)))
    fp = float(np.sum((yt == 0) & (yp == 1)))
    fn = float(np.sum((yt == 1) & (yp == 0)))

    precision = tp / max(tp + fp, 1e-12)
    recall = tp / max(tp + fn, 1e-12)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    support = int(tp + fn)

    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "support": support,
    }


f1_ = f1_score


def cheatsheet() -> str:
    return "f1_score({}) -> F1 score, precision, and recall."
