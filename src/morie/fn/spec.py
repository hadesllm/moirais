"""Specificity."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import DiagnosticResult


def specificity_dx(y_true, y_pred, *, alpha: float = 0.05) -> DiagnosticResult:
    """Specificity = TN / (TN + FP). Wilson CI.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0/1).
    y_pred : array-like
        Predicted binary labels (0/1).
    alpha : float
        Significance level for CI. Default 0.05.

    Returns
    -------
    DiagnosticResult
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    n = tn + fp
    sp = tn / n if n > 0 else 0.0
    z = stats.norm.ppf(1 - alpha / 2)
    if n > 0:
        denom = 1 + z**2 / n
        center = (sp + z**2 / (2 * n)) / denom
        hw = z * np.sqrt(sp * (1 - sp) / n + z**2 / (4 * n**2)) / denom
    else:
        center = hw = 0
    return DiagnosticResult(
        name="Specificity",
        estimate=float(sp),
        ci_lower=max(0, center - hw),
        ci_upper=min(1, center + hw),
        n=n,
        extra={"tn": tn, "fp": fp},
    )


spec = specificity_dx


def cheatsheet() -> str:
    return "specificity_dx({}) -> Specificity."
