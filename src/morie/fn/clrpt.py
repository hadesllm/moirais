# morie.fn -- function file (hadesllm/morie)
"""Classification report (all metrics per class)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "We suffer more often in imagination than in reality. -- Seneca"


def classification_report(y_true, y_pred, *, labels=None, **kwargs) -> DescriptiveResult:
    """
    Generate a classification report with per-class precision, recall, F1, and support.

    :param y_true: array-like of true labels.
    :param y_pred: array-like of predicted labels.
    :param labels: Optional ordered list of labels. Auto-detected if None.
    :return: DescriptiveResult with per-class and aggregate metrics.

    References
    ----------
    Sokolova M, Lapalme G (2009). A systematic analysis of performance
        measures for classification tasks. *Information Processing & Management*.
    """
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred must have same length.")
    if len(yt) == 0:
        raise ValueError("Inputs must not be empty.")
    if labels is None:
        labels = sorted(set(np.concatenate([np.unique(yt), np.unique(yp)])))
    labels = list(labels)
    report = {}
    macro_p, macro_r, macro_f1 = [], [], []
    weighted_p, weighted_r, weighted_f1 = 0.0, 0.0, 0.0
    total_support = len(yt)
    for lab in labels:
        tp = int(np.sum((yt == lab) & (yp == lab)))
        fp = int(np.sum((yt != lab) & (yp == lab)))
        fn = int(np.sum((yt == lab) & (yp != lab)))
        sup = int(np.sum(yt == lab))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        report[str(lab)] = {"precision": prec, "recall": rec, "f1": f1, "support": sup}
        macro_p.append(prec)
        macro_r.append(rec)
        macro_f1.append(f1)
        weighted_p += prec * sup
        weighted_r += rec * sup
        weighted_f1 += f1 * sup
    accuracy = int(np.sum(yt == yp)) / len(yt)
    report["macro_avg"] = {
        "precision": float(np.mean(macro_p)),
        "recall": float(np.mean(macro_r)),
        "f1": float(np.mean(macro_f1)),
    }
    report["weighted_avg"] = {
        "precision": weighted_p / total_support if total_support > 0 else 0.0,
        "recall": weighted_r / total_support if total_support > 0 else 0.0,
        "f1": weighted_f1 / total_support if total_support > 0 else 0.0,
    }
    report["accuracy"] = accuracy
    return DescriptiveResult(
        name="classification_report",
        value=accuracy,
        extra=report,
    )


clrpt = classification_report


def cheatsheet() -> str:
    return "classification_report({}) -> Classification report (per-class metrics)."
