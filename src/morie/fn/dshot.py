# morie.fn — function file (hadesllm/morie)
"""Precision at K metric. 'I never miss.' -- Deadshot"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def precision_at_k(
    y_true: np.ndarray | list,
    y_scores: np.ndarray | list,
    *,
    k: int = 10,
) -> DescriptiveResult:
    """Precision at K: proportion of true positives in the top K predictions.

    Parameters
    ----------
    y_true : array
        Binary ground truth labels (1 = relevant).
    y_scores : array
        Predicted scores (higher = more likely positive).
    k : int
        Number of top predictions to evaluate.

    Returns
    -------
    DescriptiveResult
        ``value`` = precision at K.
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_scores = np.asarray(y_scores, dtype=float).ravel()
    if len(y_true) != len(y_scores):
        raise ValueError("y_true and y_scores must have the same length")
    n = len(y_true)
    if n == 0:
        raise ValueError("Empty arrays")
    k = min(k, n)
    if k < 1:
        raise ValueError("k must be >= 1")
    order = np.argsort(-y_scores)
    top_k_labels = y_true[order[:k]]
    p_at_k = float(np.sum(top_k_labels)) / k
    ap = 0.0
    tp = 0
    for i, idx in enumerate(order):
        if y_true[idx] == 1:
            tp += 1
            ap += tp / (i + 1)
    n_pos = int(y_true.sum())
    ap = ap / n_pos if n_pos > 0 else 0.0
    return DescriptiveResult(
        name=f"Precision@{k}",
        value=p_at_k,
        extra={
            "k": k,
            "n": n,
            "n_positive": n_pos,
            "precision_at_k": p_at_k,
            "average_precision": round(ap, 4),
            "recall_at_k": float(np.sum(top_k_labels)) / n_pos if n_pos > 0 else 0.0,
        },
    )


dshot = precision_at_k


def cheatsheet() -> str:
    return "precision_at_k({}) -> Precision at K metric. 'I never miss.' -- Deadshot"
