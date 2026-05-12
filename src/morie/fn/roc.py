# morie.fn -- function file (hadesllm/morie)
"""ROC curve and AUC."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import DiagnosticResult


def roc_auc(y_true, y_score) -> DiagnosticResult:
    """ROC curve + AUC (trapezoidal rule).

    Parameters
    ----------
    y_true : array-like
        True binary labels (0/1).
    y_score : array-like
        Predicted scores / probabilities.

    Returns
    -------
    DiagnosticResult
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    mask = np.isfinite(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    thresholds = np.unique(y_score)[::-1]
    tpr_list, fpr_list = [0.0], [0.0]
    P, N = int(np.sum(y_true == 1)), int(np.sum(y_true == 0))
    for thresh in thresholds:
        pred = (y_score >= thresh).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        tpr_list.append(float(tp / P) if P > 0 else 0)
        fpr_list.append(float(fp / N) if N > 0 else 0)
    tpr_list.append(1.0)
    fpr_list.append(1.0)
    fpr = np.array(fpr_list)
    tpr = np.array(tpr_list)
    order = np.argsort(fpr)
    fpr, tpr = fpr[order], tpr[order]
    auc_val = float(np.trapezoid(tpr, fpr))
    # DeLong SE approximation
    se = np.sqrt(auc_val * (1 - auc_val) / min(P, N)) if min(P, N) > 0 else 0
    z = stats.norm.ppf(0.975)
    return DiagnosticResult(
        name="ROC-AUC",
        estimate=auc_val,
        ci_lower=max(0, auc_val - z * se),
        ci_upper=min(1, auc_val + z * se),
        n=len(y_true),
        extra={"fpr": fpr.tolist(), "tpr": tpr.tolist(), "se": float(se)},
    )


roc = roc_auc


def cheatsheet() -> str:
    return "roc_auc({}) -> ROC curve and AUC."
