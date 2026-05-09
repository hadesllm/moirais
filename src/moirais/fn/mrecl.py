# moirais.fn — function file (hadesllm/moirais)
"""Macro/micro/weighted average recall (multi-class)."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes
def multiclass_recall(y_true, y_pred, *, average="macro", labels=None, **kwargs) -> ESRes:
    """
    Compute recall averaged over multiple classes.

    Averaging modes:
    - macro: unweighted mean of per-class recall
    - micro: TP_total / (TP_total + FN_total)
    - weighted: weighted mean by class support

    :param y_true: array-like of true labels.
    :param y_pred: array-like of predicted labels.
    :param average: 'macro', 'micro', or 'weighted'. Default 'macro'.
    :param labels: Optional ordered list of labels. Auto-detected if None.
    :return: ESRes with averaged recall.
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
    per_class_rec = []
    supports = []
    tp_total, fn_total = 0, 0
    for lab in labels:
        tp = int(np.sum((yt == lab) & (yp == lab)))
        fn = int(np.sum((yt == lab) & (yp != lab)))
        sup = int(np.sum(yt == lab))
        per_class_rec.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        supports.append(sup)
        tp_total += tp
        fn_total += fn
    if average == "micro":
        result = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    elif average == "weighted":
        total_sup = sum(supports)
        result = sum(r * s for r, s in zip(per_class_rec, supports)) / total_sup if total_sup > 0 else 0.0
    else:
        result = float(np.mean(per_class_rec)) if per_class_rec else 0.0
    return ESRes(
        measure=f"recall_{average}",
        estimate=result,
        n=len(yt),
        extra={"per_class": dict(zip([str(l) for l in labels], per_class_rec)), "average": average},
    )


mrecl = multiclass_recall


def cheatsheet() -> str:
    return "multiclass_recall({}) -> Macro/micro/weighted recall."
