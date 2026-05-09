# moirais.fn — function file (hadesllm/moirais)
"""Confusion matrix and derived metrics."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def confusion_matrix(
    y_true: Union[np.ndarray, Any],
    y_pred: Union[np.ndarray, Any],
) -> dict[str, Any]:
    """Compute confusion matrix and derived classification metrics.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Ground-truth binary labels (0/1).
    y_pred : array-like of shape (n,)
        Predicted binary labels (0/1).

    Returns
    -------
    dict
        matrix (2x2 ndarray), tp, fp, fn, tn, accuracy, precision, recall, f1.

    References
    ----------
    Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition
        Letters*, 27(8), 861-874.
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("y_true and y_pred must have same length.")

    tp = int(np.sum((yt == 1) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    matrix = np.array([[tn, fp], [fn, tp]])

    return {
        "matrix": matrix,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


cm = confusion_matrix


def cheatsheet() -> str:
    return "confusion_matrix({}) -> Confusion matrix and derived metrics."
