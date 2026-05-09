# moirais.fn — function file (hadesllm/moirais)
"""Gradient boosting classifier (pure NumPy)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult
from ._richresult import RichResult

_QUOTE = "The ability to speak does not make you intelligent."


def gbm_classify_bio(X_train, y_train, X_test, n_estimators=100, lr=0.1, max_depth=3, **kwargs) -> DescriptiveResult:
    """Gradient boosted classification using log-loss and regression stumps.

    Binary classification via logistic loss. Each stage fits a shallow
    regression tree to the negative gradient (residuals).

    Parameters
    ----------
    X_train : array-like of shape (n_train, p)
    y_train : array-like of shape (n_train,)
    X_test : array-like of shape (n_test, p)
    n_estimators : int
        Boosting rounds (default 100).
    lr : float
        Learning rate / shrinkage (default 0.1).
    max_depth : int
        Depth per tree (default 3).

    Returns
    -------
    DescriptiveResult

    References
    ----------
    Friedman, J.H. (2001). Greedy function approximation: A gradient
        boosting machine. *Ann. Statist.*, 29(5), 1189--1232.
    """
    X_tr = np.asarray(X_train, dtype=float)
    X_te = np.asarray(X_test, dtype=float)
    y = np.asarray(y_train).ravel()
    classes = np.unique(y)
    y_bin = np.where(y == classes[1], 1.0, 0.0)

    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _fit_stump(X, residuals, depth=0):
        if depth >= max_depth or len(residuals) <= 1:
            return RichResult(payload={"leaf": True, "val": float(np.mean(residuals))})
        best_gain = -1.0
        best = None
        var_r = np.var(residuals)
        ns = len(residuals)
        if var_r < 1e-12:
            return RichResult(payload={"leaf": True, "val": float(np.mean(residuals))})
        for j in range(X.shape[1]):
            vals = np.unique(X[:, j])
            if len(vals) > 20:
                vals = np.percentile(X[:, j], np.linspace(0, 100, 22)[1:-1])
            for t in vals:
                left = X[:, j] <= t
                nl = left.sum()
                nr = ns - nl
                if nl == 0 or nr == 0:
                    continue
                g = var_r - (nl / ns * np.var(residuals[left]) + nr / ns * np.var(residuals[~left]))
                if g > best_gain:
                    best_gain = g
                    best = (j, float(t), left)
        if best is None or best_gain <= 0:
            return RichResult(payload={"leaf": True, "val": float(np.mean(residuals))})
        j, t, left = best
        return {
            "leaf": False,
            "feature": j,
            "threshold": t,
            "left": _fit_stump(X[left], residuals[left], depth + 1),
            "right": _fit_stump(X[~left], residuals[~left], depth + 1),
        }

    def _pred_stump(x, node):
        if node["leaf"]:
            return node["val"]
        return _pred_stump(x, node["left"] if x[node["feature"]] <= node["threshold"] else node["right"])

    F_train = np.zeros(len(X_tr))
    trees = []

    for _ in range(n_estimators):
        prob = _sigmoid(F_train)
        residuals = y_bin - prob
        tree = _fit_stump(X_tr, residuals)
        updates = np.array([_pred_stump(X_tr[i], tree) for i in range(len(X_tr))])
        F_train += lr * updates
        trees.append(tree)

    F_test = np.zeros(len(X_te))
    for tree in trees:
        F_test += lr * np.array([_pred_stump(X_te[i], tree) for i in range(len(X_te))])

    prob_test = _sigmoid(F_test)
    pred_labels = np.where(prob_test >= 0.5, classes[1], classes[0])
    train_prob = _sigmoid(F_train)
    train_acc = float(np.mean((train_prob >= 0.5) == y_bin))

    return DescriptiveResult(
        name="gbm_classify_bio",
        value=train_acc,
        extra={
            "predictions": pred_labels,
            "probabilities": prob_test,
            "train_accuracy": train_acc,
            "n_estimators": n_estimators,
            "learning_rate": lr,
        },
    )


gbcls = gbm_classify_bio


def cheatsheet() -> str:
    return "gbm_classify_bio({}) -> Gradient boosting classifier (pure NumPy)."
