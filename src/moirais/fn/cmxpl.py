# moirais.fn — function file (hadesllm/moirais)
"""Confusion matrix with per-class metrics."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Truly wonderful, the mind of a child is."


def confusion_plot(y_true, y_pred, **kwargs) -> DescriptiveResult:
    """Compute confusion matrix and per-class precision/recall/F1.

    Parameters
    ----------
    y_true : array-like of shape (n,)
    y_pred : array-like of shape (n,)

    Returns
    -------
    DescriptiveResult
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    classes = np.unique(np.concatenate([y_true, y_pred]))
    k = len(classes)
    cm = np.zeros((k, k), dtype=int)
    class_map = {c: i for i, c in enumerate(classes)}

    for t, p in zip(y_true, y_pred):
        cm[class_map[t], class_map[p]] += 1

    per_class = {}
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1, "support": int(cm[i, :].sum())}

    accuracy = float(np.trace(cm) / cm.sum())

    return DescriptiveResult(
        name="confusion_plot",
        value=accuracy,
        extra={
            "confusion_matrix": cm,
            "classes": classes,
            "per_class": per_class,
            "accuracy": accuracy,
        },
    )


cmxpl = confusion_plot


def cheatsheet() -> str:
    return "confusion_plot({}) -> Confusion matrix with per-class metrics."
