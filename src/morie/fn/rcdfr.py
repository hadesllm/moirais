# morie.fn -- function file (hadesllm/morie)
"""Fairness metrics for recidivism predictions."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import DescriptiveResult


def recidivism_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group: np.ndarray,
) -> DescriptiveResult:
    """Fairness metrics: FPR/FNR ratio across groups.

    Parameters
    ----------
    y_true : ndarray
        True binary outcomes.
    y_pred : ndarray
        Predicted binary outcomes.
    group : ndarray
        Group membership labels.

    Returns
    -------
    DescriptiveResult
        extra contains per-group FPR, FNR, and ratios.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    group = np.asarray(group)
    groups = np.unique(group)
    metrics = {}
    for g in groups:
        mask = group == g
        yt, yp = y_true[mask], y_pred[mask]
        fp = np.sum((yp == 1) & (yt == 0))
        fn = np.sum((yp == 0) & (yt == 1))
        tn = np.sum((yp == 0) & (yt == 0))
        tp = np.sum((yp == 1) & (yt == 1))
        fpr = fp / max(fp + tn, 1)
        fnr = fn / max(fn + tp, 1)
        metrics[str(g)] = {"fpr": float(fpr), "fnr": float(fnr), "n": int(np.sum(mask))}
    fpr_values = [m["fpr"] for m in metrics.values()]
    fpr_ratio = max(fpr_values) / max(min(fpr_values), 1e-10) if len(fpr_values) > 1 else 1.0
    return DescriptiveResult(
        name="recidivism_fairness",
        value=float(fpr_ratio),
        extra={"group_metrics": metrics, "fpr_disparity_ratio": float(fpr_ratio)},
    )


rcdfr = recidivism_fairness


def cheatsheet() -> str:
    return "recidivism_fairness({}) -> Fairness metrics for recidivism predictions."
