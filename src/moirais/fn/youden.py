"""Youden's J index."""

from __future__ import annotations

import numpy as np

from ._containers import DiagnosticResult


def youdens_j(y_true, y_score) -> DiagnosticResult:
    """Youden's J = max(sensitivity + specificity - 1). Finds optimal threshold.

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
    thresholds = np.unique(y_score)
    best_j, best_thresh = -1.0, 0.0
    P, N = int(np.sum(y_true == 1)), int(np.sum(y_true == 0))
    for thresh in thresholds:
        pred = (y_score >= thresh).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        tn = np.sum((pred == 0) & (y_true == 0))
        sens_val = float(tp / P) if P > 0 else 0
        spec_val = float(tn / N) if N > 0 else 0
        j = sens_val + spec_val - 1
        if j > best_j:
            best_j, best_thresh = j, float(thresh)
    return DiagnosticResult(
        name="Youden's J",
        estimate=float(best_j),
        n=len(y_true),
        extra={"optimal_threshold": best_thresh, "j_index": float(best_j)},
    )


youden = youdens_j


def cheatsheet() -> str:
    return "youdens_j({}) -> Youden's J index."
