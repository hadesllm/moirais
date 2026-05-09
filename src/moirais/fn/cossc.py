# moirais.fn — function file (hadesllm/moirais)
"""Cost-sensitive evaluation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Your overconfidence is your weakness."


def cost_sensitive(y_true, y_pred, cost_matrix=None, **kwargs) -> DescriptiveResult:
    """Cost-sensitive evaluation of classifier predictions.

    Parameters
    ----------
    y_true : array-like of shape (n,)
    y_pred : array-like of shape (n,)
    cost_matrix : array-like of shape (k, k) or None
        Cost[i, j] = cost of predicting class j when truth is class i.
        Default: 0 on diagonal, 1 off-diagonal (0-1 loss),
        with FN cost = 2 for binary (asymmetric).

    Returns
    -------
    DescriptiveResult
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    classes = np.unique(np.concatenate([y_true, y_pred]))
    k = len(classes)
    class_map = {c: i for i, c in enumerate(classes)}

    if cost_matrix is None:
        cost_matrix = np.ones((k, k)) - np.eye(k)
        if k == 2:
            cost_matrix[1, 0] = 2.0
    cost_matrix = np.asarray(cost_matrix, dtype=float)

    total_cost = 0.0
    for t, p in zip(y_true, y_pred):
        total_cost += cost_matrix[class_map[t], class_map[p]]

    avg_cost = float(total_cost / len(y_true)) if len(y_true) > 0 else 0.0

    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[class_map[t], class_map[p]] += 1

    return DescriptiveResult(
        name="cost_sensitive",
        value=avg_cost,
        extra={
            "total_cost": float(total_cost),
            "average_cost": avg_cost,
            "cost_matrix": cost_matrix,
            "confusion_matrix": cm,
            "classes": classes,
        },
    )


cossc = cost_sensitive


def cheatsheet() -> str:
    return "cost_sensitive({}) -> Cost-sensitive evaluation."
