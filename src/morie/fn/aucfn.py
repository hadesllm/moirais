# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Area under ROC curve."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Never tell me the odds."


def auc_compute(y_true, y_scores, **kwargs) -> DescriptiveResult:
    """Compute area under the ROC curve via trapezoidal rule.

    .. math::

        \\text{AUC} = \\int_0^1 \\text{TPR}(t)\\; d\\text{FPR}(t)

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary labels (0/1).
    y_scores : array-like of shape (n,)
        Predicted scores.

    Returns
    -------
    DescriptiveResult
    """
    y_true = np.asarray(y_true).ravel()
    y_scores = np.asarray(y_scores, dtype=float).ravel()

    desc = np.argsort(-y_scores)
    y_sorted = y_true[desc]
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    if P == 0 or N == 0:
        raise ValueError("Need both classes present.")

    tp = 0
    fp = 0
    tpr_list = [0.0]
    fpr_list = [0.0]
    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / P)
        fpr_list.append(fp / N)

    fpr = np.array(fpr_list)
    tpr = np.array(tpr_list)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = float(_trapz(tpr, fpr))

    return DescriptiveResult(
        name="auc_compute",
        value=auc,
        extra={"auc": auc, "n_positive": int(P), "n_negative": int(N)},
    )


aucfn = auc_compute


def cheatsheet() -> str:
    return "auc_compute({}) -> Area under ROC curve."
