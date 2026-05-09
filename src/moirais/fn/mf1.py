# moirais.fn — function file (hadesllm/moirais)
"""Macro/micro/weighted F1 score (multi-class)."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes

_QUOTE = "In my experience there is no such thing as luck. --"


def multiclass_f1(y_true, y_pred, *, average="macro", labels=None, **kwargs) -> ESRes:
    """
    Compute F1 score averaged over multiple classes.

    Averaging modes:
    - macro: unweighted mean of per-class F1
    - micro: 2*TP_total / (2*TP_total + FP_total + FN_total)
    - weighted: weighted mean by class support

    :param y_true: array-like of true labels.
    :param y_pred: array-like of predicted labels.
    :param average: 'macro', 'micro', or 'weighted'. Default 'macro'.
    :param labels: Optional ordered list of labels. Auto-detected if None.
    :return: ESRes with averaged F1.
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
    per_class_f1 = []
    supports = []
    tp_total, fp_total, fn_total = 0, 0, 0
    for lab in labels:
        tp = int(np.sum((yt == lab) & (yp == lab)))
        fp = int(np.sum((yt != lab) & (yp == lab)))
        fn = int(np.sum((yt == lab) & (yp != lab)))
        sup = int(np.sum(yt == lab))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class_f1.append(f1)
        supports.append(sup)
        tp_total += tp
        fp_total += fp
        fn_total += fn
    if average == "micro":
        result = 2 * tp_total / (2 * tp_total + fp_total + fn_total) if (2 * tp_total + fp_total + fn_total) > 0 else 0.0
    elif average == "weighted":
        total_sup = sum(supports)
        result = sum(f * s for f, s in zip(per_class_f1, supports)) / total_sup if total_sup > 0 else 0.0
    else:
        result = float(np.mean(per_class_f1)) if per_class_f1 else 0.0
    return ESRes(
        measure=f"f1_{average}",
        estimate=result,
        n=len(yt),
        extra={"per_class": dict(zip([str(l) for l in labels], per_class_f1)), "average": average},
    )


mf1 = multiclass_f1


def cheatsheet() -> str:
    return "multiclass_f1({}) -> Macro/micro/weighted F1."
