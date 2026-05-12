# morie.fn -- function file (hadesllm/morie)
"""Precision-recall at top-k. 'You and I see things differently.' -- Hawkeye"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def precision_recall_at_k(
    y_true: np.ndarray | list,
    y_scores: np.ndarray | list[float],
    *,
    k: int = 10,
) -> DescriptiveResult:
    """Compute precision and recall at top-k ranked predictions.

    Parameters
    ----------
    y_true : array-like
        Binary ground-truth labels (0/1).
    y_scores : array-like
        Predicted scores (higher = more likely positive).
    k : int
        Number of top predictions to evaluate.

    Returns
    -------
    DescriptiveResult
        ``value`` dict with ``precision_at_k``, ``recall_at_k``,
        ``f1_at_k``, ``n_relevant``, ``k``.
    """
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_scores, dtype=float)
    if len(y) != len(s):
        raise ValueError("y_true and y_scores must have same length")
    if k < 1:
        raise ValueError("k must be >= 1")

    k = min(k, len(y))
    order = np.argsort(-s)
    top_k = y[order[:k]]

    n_relevant_total = int(y.sum())
    tp = int(top_k.sum())
    precision = tp / k if k > 0 else 0.0
    recall = tp / n_relevant_total if n_relevant_total > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return DescriptiveResult(
        name="precision_recall_at_k",
        value={
            "precision_at_k": float(precision),
            "recall_at_k": float(recall),
            "f1_at_k": float(f1),
            "n_relevant": n_relevant_total,
            "k": k,
        },
        extra={"n": len(y)},
    )


hkeye = precision_recall_at_k


def cheatsheet() -> str:
    return "precision_recall_at_k({}) -> Precision-recall at top-k. 'You and I see things differently"
