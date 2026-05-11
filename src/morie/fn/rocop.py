# morie.fn — function file (hadesllm/morie)
"""ROC optimal cutoff selection (Youden, closest-to-corner)."""

from __future__ import annotations

from typing import Any

import numpy as np


def roc_optimal_cutoff(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    method: str = "youden",
    cost_fp: float = 1.0,
    cost_fn: float = 1.0,
) -> dict[str, Any]:
    """Find the optimal cutoff from ROC curve analysis.

    Methods:
    - **youden**: maximizes J = Se + Sp - 1
    - **closest**: minimizes Euclidean distance to (0, 1)
    - **cost**: minimizes weighted misclassification cost

    Parameters
    ----------
    y_true : array_like
        Binary true labels (0/1).
    y_score : array_like
        Continuous predicted scores.
    method : str, default "youden"
        "youden", "closest", or "cost".
    cost_fp : float, default 1.0
        Cost of a false positive (for cost method).
    cost_fn : float, default 1.0
        Cost of a false negative (for cost method).

    Returns
    -------
    dict
        Keys: 'cutoff', 'sensitivity', 'specificity', 'youden_index',
              'auc', 'fpr_array', 'tpr_array', 'thresholds'.

    References
    ----------
    Youden, W. J. (1950). Index for rating diagnostic tests. Cancer,
    3(1), 32-35.

    Perkins, N. J. & Schisterman, E. F. (2006). The inconsistency of
    "optimal" cutpoints obtained using two criteria based on the ROC
    curve. American Journal of Epidemiology, 163(7), 670-675.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have same shape.")
    if set(np.unique(y_true)) - {0, 1}:
        raise ValueError("y_true must be binary (0/1).")

    thresholds = np.unique(y_score)
    thresholds = np.sort(thresholds)[::-1]

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    tpr = np.empty(len(thresholds))
    fpr = np.empty(len(thresholds))

    for i, th in enumerate(thresholds):
        pred = (y_score >= th).astype(int)
        tpr[i] = np.sum((pred == 1) & (y_true == 1)) / n_pos if n_pos > 0 else 0.0
        fpr[i] = np.sum((pred == 1) & (y_true == 0)) / n_neg if n_neg > 0 else 0.0

    auc = float(np.abs(np.trapezoid(tpr, fpr)))

    if method == "youden":
        j = tpr - fpr
        best = int(np.argmax(j))
    elif method == "closest":
        dist = np.sqrt(fpr**2 + (1 - tpr)**2)
        best = int(np.argmin(dist))
    elif method == "cost":
        total_cost = cost_fp * fpr * n_neg + cost_fn * (1 - tpr) * n_pos
        best = int(np.argmin(total_cost))
    else:
        raise ValueError("method must be 'youden', 'closest', or 'cost'.")

    return {
        "cutoff": float(thresholds[best]),
        "sensitivity": float(tpr[best]),
        "specificity": float(1 - fpr[best]),
        "youden_index": float(tpr[best] - fpr[best]),
        "auc": auc,
        "fpr_array": fpr,
        "tpr_array": tpr,
        "thresholds": thresholds,
    }


rocop = roc_optimal_cutoff


def cheatsheet() -> str:
    return "roc_optimal_cutoff({}) -> ROC optimal cutoff (Youden/closest/cost)."
