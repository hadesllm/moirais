# moirais.fn — function file (hadesllm/moirais)
"""Macro/micro/weighted average precision (multi-class)."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes

_QUOTE = "Difficulties strengthen the mind, as labor does the body. — Seneca"


def multiclass_precision(y_true, y_pred, *, average="macro", labels=None, **kwargs) -> ESRes:
    """
    Compute precision averaged over multiple classes.

    Averaging modes:
    - macro: unweighted mean of per-class precision
    - micro: TP_total / (TP_total + FP_total)
    - weighted: weighted mean by class support

    :param y_true: array-like of true labels.
    :param y_pred: array-like of predicted labels.
    :param average: 'macro', 'micro', or 'weighted'. Default 'macro'.
    :param labels: Optional ordered list of labels. Auto-detected if None.
    :return: ESRes with averaged precision.
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
    per_class_prec = []
    supports = []
    tp_total, fp_total = 0, 0
    for lab in labels:
        tp = int(np.sum((yt == lab) & (yp == lab)))
        fp = int(np.sum((yt != lab) & (yp == lab)))
        sup = int(np.sum(yt == lab))
        per_class_prec.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        supports.append(sup)
        tp_total += tp
        fp_total += fp
    if average == "micro":
        result = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    elif average == "weighted":
        total_sup = sum(supports)
        result = sum(p * s for p, s in zip(per_class_prec, supports)) / total_sup if total_sup > 0 else 0.0
    else:
        result = float(np.mean(per_class_prec)) if per_class_prec else 0.0
    return ESRes(
        measure=f"precision_{average}",
        estimate=result,
        n=len(yt),
        extra={"per_class": dict(zip([str(l) for l in labels], per_class_prec)), "average": average},
    )


maccr = multiclass_precision


def cheatsheet() -> str:
    return "multiclass_precision({}) -> Macro/micro/weighted precision."
