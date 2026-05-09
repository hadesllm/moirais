# moirais.fn — function file (hadesllm/moirais)
"""Multiclass averaging (macro/micro/weighted)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The greatest teacher, failure is."


def multiclass_avg(y_true, y_pred, average="macro", **kwargs) -> DescriptiveResult:
    """Compute precision, recall, F1 with macro/micro/weighted averaging.

    Parameters
    ----------
    y_true : array-like of shape (n,)
    y_pred : array-like of shape (n,)
    average : str
        "macro", "micro", or "weighted" (default "macro").

    Returns
    -------
    DescriptiveResult
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    classes = np.unique(np.concatenate([y_true, y_pred]))

    per_class = {}
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1, "support": int(np.sum(y_true == c))}

    if average == "micro":
        tp_all = sum(np.sum((y_pred == c) & (y_true == c)) for c in classes)
        fp_all = sum(np.sum((y_pred == c) & (y_true != c)) for c in classes)
        fn_all = sum(np.sum((y_pred != c) & (y_true == c)) for c in classes)
        precision = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0.0
        recall = tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    elif average == "weighted":
        total = len(y_true)
        precision = sum(per_class[c]["precision"] * per_class[c]["support"] for c in classes) / total
        recall = sum(per_class[c]["recall"] * per_class[c]["support"] for c in classes) / total
        f1 = sum(per_class[c]["f1"] * per_class[c]["support"] for c in classes) / total
    else:
        precision = np.mean([per_class[c]["precision"] for c in classes])
        recall = np.mean([per_class[c]["recall"] for c in classes])
        f1 = np.mean([per_class[c]["f1"] for c in classes])

    accuracy = float(np.mean(y_true == y_pred))

    return DescriptiveResult(
        name="multiclass_avg",
        value=float(f1),
        extra={
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": accuracy,
            "average": average,
            "per_class": per_class,
        },
    )


mcavg = multiclass_avg


def cheatsheet() -> str:
    return "multiclass_avg({}) -> Multiclass averaging (macro/micro/weighted)."
