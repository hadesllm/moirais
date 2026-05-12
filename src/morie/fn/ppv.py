# morie.fn -- function file (hadesllm/morie)
"""Positive and negative predictive values."""

from __future__ import annotations

import numpy as np

from ._containers import DiagnosticResult


def ppv_npv(y_true, y_pred) -> DiagnosticResult:
    """PPV = TP/(TP+FP), NPV = TN/(TN+FN).

    Parameters
    ----------
    y_true : array-like
        True binary labels (0/1).
    y_pred : array-like
        Predicted binary labels (0/1).

    Returns
    -------
    DiagnosticResult
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    ppv_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv_val = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return DiagnosticResult(
        name="PPV/NPV",
        estimate=float(ppv_val),
        n=len(y_true),
        extra={
            "ppv": ppv_val,
            "npv": npv_val,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        },
    )


ppv = ppv_npv


def cheatsheet() -> str:
    return "ppv_npv({}) -> Positive and negative predictive values."
