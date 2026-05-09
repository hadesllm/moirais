# moirais.fn — function file (hadesllm/moirais)
"""CART decision tree classifier (pure NumPy, Gini splitting)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult
from ._richresult import RichResult

_QUOTE = "The only true wisdom is in knowing you know nothing. — Socrates"


def decision_tree(X_train, y_train, X_test, max_depth=5, min_samples_leaf=1, **kwargs) -> DescriptiveResult:
    """CART decision tree using recursive Gini impurity splitting.

    .. math::

        \\text{Gini}(t) = 1 - \\sum_{k=1}^{K} p_k^2

    Parameters
    ----------
    X_train : array-like of shape (n_train, p)
    y_train : array-like of shape (n_train,)
    X_test : array-like of shape (n_test, p)
    max_depth : int
        Maximum tree depth (default 5).
    min_samples_leaf : int
        Minimum samples per leaf.

    Returns
    -------
    DescriptiveResult

    References
    ----------
    Breiman, L., Friedman, J., Olshen, R. & Stone, C. (1984).
        *Classification and Regression Trees*. Wadsworth.
    """
    X_tr = np.asarray(X_train, dtype=float)
    X_te = np.asarray(X_test, dtype=float)
    y = np.asarray(y_train).ravel()
    n, p = X_tr.shape
    importance = np.zeros(p)
    leaf_count = [0]

    def _gini(labels):
        if len(labels) == 0:
            return 0.0
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / len(labels)
        return 1.0 - float(np.sum(probs**2))

    def _build(indices, depth):
        labels = y[indices]
        if depth >= max_depth or len(np.unique(labels)) == 1 or len(indices) <= min_samples_leaf:
            leaf_count[0] += 1
            vals, cnts = np.unique(labels, return_counts=True)
            return RichResult(payload={"leaf": True, "class": vals[np.argmax(cnts)]})

        best_gain = -1.0
        best_feat = 0
        best_thresh = 0.0
        best_left = np.array([], dtype=int)
        best_right = np.array([], dtype=int)
        parent_gini = _gini(labels)

        for j in range(p):
            vals = np.unique(X_tr[indices, j])
            if len(vals) > 20:
                vals = np.percentile(X_tr[indices, j], np.linspace(0, 100, 22)[1:-1])
            for t in vals:
                left_mask = X_tr[indices, j] <= t
                right_mask = ~left_mask
                nl, nr = left_mask.sum(), right_mask.sum()
                if nl < min_samples_leaf or nr < min_samples_leaf:
                    continue
                gain = parent_gini - (
                    nl / len(indices) * _gini(labels[left_mask]) + nr / len(indices) * _gini(labels[right_mask])
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feat = j
                    best_thresh = float(t)
                    best_left = indices[left_mask]
                    best_right = indices[right_mask]

        if best_gain <= 0 or len(best_left) == 0 or len(best_right) == 0:
            leaf_count[0] += 1
            vals, cnts = np.unique(labels, return_counts=True)
            return RichResult(payload={"leaf": True, "class": vals[np.argmax(cnts)]})

        importance[best_feat] += best_gain * len(indices) / n
        return {
            "leaf": False,
            "feature": best_feat,
            "threshold": best_thresh,
            "left": _build(best_left, depth + 1),
            "right": _build(best_right, depth + 1),
        }

    tree = _build(np.arange(n), 0)

    def _predict_one(x, node):
        if node["leaf"]:
            return node["class"]
        if x[node["feature"]] <= node["threshold"]:
            return _predict_one(x, node["left"])
        return _predict_one(x, node["right"])

    preds_train = np.array([_predict_one(X_tr[i], tree) for i in range(n)])
    preds_test = np.array([_predict_one(X_te[i], tree) for i in range(len(X_te))])
    train_acc = float(np.mean(preds_train == y))
    imp_sum = importance.sum()
    if imp_sum > 0:
        importance /= imp_sum

    return DescriptiveResult(
        name="decision_tree",
        value=train_acc,
        extra={
            "predictions": preds_test,
            "train_predictions": preds_train,
            "feature_importance": importance,
            "n_leaves": leaf_count[0],
            "train_accuracy": train_acc,
            "max_depth": max_depth,
        },
    )


dtree = decision_tree


def cheatsheet() -> str:
    return "decision_tree({}) -> CART decision tree classifier (pure NumPy, Gini splitting)."
