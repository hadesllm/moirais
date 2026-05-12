"""Confusion matrix with precision / recall / F1 (per class + macro)."""
import numpy as np

from ._richresult import RichResult

__all__ = ["confusion_matrix_metrics"]


def confusion_matrix_metrics(y_true, y_pred, *, labels=None):
    """Confusion matrix + classification report via sklearn.metrics.

    F1 = 2 P R / (P + R), P = TP / (TP + FP), R = TP / (TP + FN).
    Works for binary or multiclass.  Returns per-class precision/recall/F1
    plus accuracy and macro/weighted averages.

    Parameters
    ----------
    y_true : array-like (n,).
    y_pred : array-like (n,).
    labels : array-like or None
        Class label ordering; defaults to sorted unique of (y_true ∪ y_pred).

    Returns
    -------
    RichResult with payload: estimate (accuracy), confusion_matrix, labels,
    precision, recall, f1, macro_f1, weighted_f1, accuracy, n, method.
    """
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, precision_recall_fscore_support,
    )

    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    if labels is None:
        labels = np.unique(np.concatenate([yt, yp]))
    cm = confusion_matrix(yt, yp, labels=labels)
    p, r, f1, _ = precision_recall_fscore_support(
        yt, yp, labels=labels, zero_division=0,
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        yt, yp, labels=labels, average="macro", zero_division=0,
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        yt, yp, labels=labels, average="weighted", zero_division=0,
    )
    acc = float(accuracy_score(yt, yp))
    return RichResult(payload={
        "estimate": acc,
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "labels": list(labels),
        "precision": p.tolist(),
        "recall": r.tolist(),
        "f1": f1.tolist(),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_w),
        "n": int(len(yt)),
        "method": "Confusion matrix + precision/recall/F1",
    })


def cheatsheet():
    return "confm: confusion matrix with precision/recall/F1"


# CANONICAL TEST
if __name__ == "__main__":
    y_true = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1]
    y_pred = [0, 1, 1, 1, 0, 0, 1, 0, 1, 1]
    r = confusion_matrix_metrics(y_true, y_pred)
    print("confusion matrix:", r.confusion_matrix)
    print("precision:", r.precision)
    print("recall:", r.recall)
    print("f1:", r.f1)
    print("accuracy:", r.accuracy, "  macro F1:", r.macro_f1)
