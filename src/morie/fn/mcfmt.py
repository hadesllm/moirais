# morie.fn -- function file (hadesllm/morie)
"""Multi-class confusion matrix."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "There are always two sides. -- Count Dooku"


def multiclass_confusion_matrix(y_true, y_pred, *, labels=None, **kwargs) -> DescriptiveResult:
    """
    Compute a multi-class confusion matrix.

    :param y_true: array-like of true class labels.
    :param y_pred: array-like of predicted class labels.
    :param labels: Optional ordered list of class labels. Auto-detected if None.
    :return: DescriptiveResult with confusion matrix and per-class counts.

    References
    ----------
    Stehman SV (1997). Selecting and interpreting measures of thematic
        classification accuracy. *Remote Sensing of Environment*, 62(1).
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
    k = len(labels)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(yt, yp):
        i = label_to_idx.get(t)
        j = label_to_idx.get(p)
        if i is not None and j is not None:
            cm[i, j] += 1
    per_class = {}
    for idx, lab in enumerate(labels):
        tp = cm[idx, idx]
        fp = int(cm[:, idx].sum() - tp)
        fn = int(cm[idx, :].sum() - tp)
        tn = int(cm.sum() - tp - fp - fn)
        per_class[str(lab)] = {"tp": int(tp), "fp": fp, "fn": fn, "tn": tn}
    return DescriptiveResult(
        name="multiclass_confusion_matrix",
        value=cm.tolist(),
        extra={"labels": labels, "per_class": per_class, "n": len(yt)},
    )


mcfmt = multiclass_confusion_matrix


def cheatsheet() -> str:
    return "multiclass_confusion_matrix({}) -> Multi-class confusion matrix."
