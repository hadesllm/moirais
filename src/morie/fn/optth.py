# morie.fn -- function file (hadesllm/morie)
"""Optimal threshold selection (Youden's J or cost-based)."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes

_QUOTE = "The right choice is the hardest one. -- Ahsoka"


def optimal_threshold(y_true, y_score, *, method="youden", fp_cost=1.0, fn_cost=1.0, **kwargs) -> ESRes:
    """
    Find the optimal classification threshold.

    Methods:
    - 'youden': maximizes Youden's J = sensitivity + specificity - 1
    - 'cost': minimizes fp_cost * FP + fn_cost * FN

    :param y_true: array-like of true binary labels (0/1).
    :param y_score: array-like of predicted scores/probabilities.
    :param method: 'youden' or 'cost'. Default 'youden'.
    :param fp_cost: Cost of a false positive (for method='cost'). Default 1.0.
    :param fn_cost: Cost of a false negative (for method='cost'). Default 1.0.
    :return: ESRes with optimal threshold and associated metrics.

    References
    ----------
    Youden WJ (1950). Index for rating diagnostic tests. *Cancer*, 3(1), 32-35.
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    ys = np.asarray(y_score, dtype=float).ravel()
    if len(yt) != len(ys):
        raise ValueError("y_true and y_score must have same length.")
    if len(yt) == 0:
        raise ValueError("Inputs must not be empty.")
    total_pos = int(np.sum(yt == 1))
    total_neg = int(np.sum(yt == 0))
    if total_pos == 0 or total_neg == 0:
        raise ValueError("y_true must contain both positive and negative samples.")
    thresholds = np.unique(ys)
    best_thresh = float(thresholds[0])
    best_score = -np.inf
    best_sens = 0.0
    best_spec = 0.0
    for t in thresholds:
        yp = (ys >= t).astype(float)
        tp = np.sum((yt == 1) & (yp == 1))
        fn = np.sum((yt == 1) & (yp == 0))
        fp = np.sum((yt == 0) & (yp == 1))
        tn = np.sum((yt == 0) & (yp == 0))
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        if method == "youden":
            score = sens + spec - 1.0
        else:
            score = -(fp_cost * fp + fn_cost * fn)
        if score > best_score:
            best_score = score
            best_thresh = float(t)
            best_sens = float(sens)
            best_spec = float(spec)
    return ESRes(
        measure=f"optimal_threshold_{method}",
        estimate=best_thresh,
        n=len(yt),
        extra={
            "method": method,
            "sensitivity": best_sens,
            "specificity": best_spec,
            "youdens_j": best_sens + best_spec - 1.0,
        },
    )


optth = optimal_threshold


def cheatsheet() -> str:
    return "optimal_threshold({}) -> Optimal threshold selection."
