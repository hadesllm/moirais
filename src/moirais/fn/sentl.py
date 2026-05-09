# moirais.fn — function file (hadesllm/moirais)
"""He who is brave is free. — Seneca"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def detection_metrics(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    *,
    threshold: float = 0.5,
) -> DescriptiveResult:
    """Compute precision, recall, F1, and average precision for detection.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels (0 or 1).
    y_pred : array-like
        Predicted scores or probabilities.
    threshold : float
        Decision threshold for converting scores to binary predictions.

    Returns
    -------
    DescriptiveResult
        ``value`` is the F1 score; ``extra`` has precision, recall, AP,
        and confusion matrix counts.
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred must have the same length")
    if len(yt) == 0:
        raise ValueError("Empty input")

    binary = (yp >= threshold).astype(float)
    tp = int(np.sum((binary == 1) & (yt == 1)))
    fp = int(np.sum((binary == 1) & (yt == 0)))
    fn = int(np.sum((binary == 0) & (yt == 1)))
    tn = int(np.sum((binary == 0) & (yt == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    order = np.argsort(-yp)
    yt_sorted = yt[order]
    cum_tp = np.cumsum(yt_sorted)
    cum_fp = np.cumsum(1 - yt_sorted)
    prec_at_k = cum_tp / (cum_tp + cum_fp)
    rec_at_k = cum_tp / max(1, yt.sum())
    ap = float(np.sum(prec_at_k * yt_sorted) / max(1, yt.sum()))

    return DescriptiveResult(
        name="Detection Metrics",
        value=round(f1, 4),
        extra={
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "average_precision": round(ap, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "threshold": threshold,
            "n": len(yt),
        },
    )


sentl = detection_metrics


def cheatsheet() -> str:
    return "detection_metrics({}) -> Object detection precision/recall. 'They are coming.' -- Mor"
