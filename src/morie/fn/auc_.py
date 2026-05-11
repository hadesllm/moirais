# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Area Under the ROC Curve (AUC) via trapezoidal integration."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def auc_score(
    y_true: Union[np.ndarray, Any],
    y_score: Union[np.ndarray, Any],
) -> float:
    """Compute Area Under the ROC Curve via trapezoidal rule.

    Pure NumPy — no sklearn required.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary ground-truth labels (0/1).
    y_score : array-like of shape (n,)
        Predicted probabilities or decision values.

    Returns
    -------
    float
        AUC value in [0, 1].

    Raises
    ------
    ValueError
        If only one class is present in y_true.

    References
    ----------
    Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition
        Letters*, 27(8), 861-874. doi:10.1016/j.patrec.2005.10.010
    """
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_s = np.asarray(y_score, dtype=float).ravel()
    if y_t.shape[0] != y_s.shape[0]:
        raise ValueError("y_true and y_score must have same length.")
    if len(np.unique(y_t)) < 2:
        raise ValueError("AUC requires both classes present in y_true.")

    # Sort by decreasing score
    order = np.argsort(-y_s)
    y_sorted = y_t[order]

    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos

    tpr_prev, fpr_prev = 0.0, 0.0
    tp, fp = 0.0, 0.0
    auc_val = 0.0

    prev_score = y_s[order[0]] + 1  # sentinel
    for i in range(len(y_sorted)):
        if y_s[order[i]] != prev_score:
            tpr = tp / n_pos
            fpr = fp / n_neg
            auc_val += 0.5 * (fpr - fpr_prev) * (tpr + tpr_prev)
            tpr_prev = tpr
            fpr_prev = fpr
            prev_score = y_s[order[i]]
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1

    # Final point
    tpr = tp / n_pos
    fpr = fp / n_neg
    auc_val += 0.5 * (fpr - fpr_prev) * (tpr + tpr_prev)

    return float(auc_val)


auc_ = auc_score


def cheatsheet() -> str:
    return "auc_score({}) -> Area Under the ROC Curve (AUC) via trapezoidal integration."
