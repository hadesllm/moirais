# morie.fn — function file (hadesllm/morie)
"""Random forest classifier (pure NumPy)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult
from ._richresult import RichResult

_QUOTE = "Mastering others is strength; mastering yourself is true power. — Lao Tzu"


def rf_classify_bio(
    X_train, y_train, X_test, n_trees=100, max_depth=5, max_features=None, **kwargs
) -> DescriptiveResult:
    """Random forest classifier with feature sub-sampling at each split.

    Each tree sees a bootstrap sample and considers sqrt(p) random features
    at every split.

    Parameters
    ----------
    X_train : array-like of shape (n_train, p)
    y_train : array-like of shape (n_train,)
    X_test : array-like of shape (n_test, p)
    n_trees : int
        Number of trees (default 100).
    max_depth : int
        Max depth per tree (default 5).
    max_features : int or None
        Features per split. Default: sqrt(p).

    Returns
    -------
    DescriptiveResult

    References
    ----------
    Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5--32.
    """
    X_tr = np.asarray(X_train, dtype=float)
    X_te = np.asarray(X_test, dtype=float)
    y = np.asarray(y_train).ravel()
    n, p = X_tr.shape
    if max_features is None:
        max_features = max(1, int(np.sqrt(p)))
    rng = np.random.default_rng(kwargs.get("seed", 42))

    def _gini(labels):
        if len(labels) == 0:
            return 0.0
        _, c = np.unique(labels, return_counts=True)
        pr = c / len(labels)
        return 1.0 - float(np.sum(pr**2))

    def _build(X_sub, y_sub, depth):
        if depth >= max_depth or len(np.unique(y_sub)) == 1 or len(y_sub) <= 1:
            vals, cnts = np.unique(y_sub, return_counts=True)
            return RichResult(payload={"leaf": True, "class": vals[np.argmax(cnts)]})
        feats = rng.choice(X_sub.shape[1], size=min(max_features, X_sub.shape[1]), replace=False)
        best_gain = -1.0
        best = None
        pg = _gini(y_sub)
        ns = len(y_sub)
        for j in feats:
            vals = np.unique(X_sub[:, j])
            if len(vals) > 20:
                vals = np.percentile(X_sub[:, j], np.linspace(0, 100, 22)[1:-1])
            for t in vals:
                left = X_sub[:, j] <= t
                nl, nr = left.sum(), ns - left.sum()
                if nl == 0 or nr == 0:
                    continue
                g = pg - (nl / ns * _gini(y_sub[left]) + nr / ns * _gini(y_sub[~left]))
                if g > best_gain:
                    best_gain = g
                    best = (j, float(t), left)
        if best is None or best_gain <= 0:
            vals, cnts = np.unique(y_sub, return_counts=True)
            return RichResult(payload={"leaf": True, "class": vals[np.argmax(cnts)]})
        j, t, left = best
        return {
            "leaf": False,
            "feature": j,
            "threshold": t,
            "left": _build(X_sub[left], y_sub[left], depth + 1),
            "right": _build(X_sub[~left], y_sub[~left], depth + 1),
        }

    def _pred(x, node):
        if node["leaf"]:
            return node["class"]
        return _pred(x, node["left"] if x[node["feature"]] <= node["threshold"] else node["right"])

    trees = []
    for _ in range(n_trees):
        idx = rng.integers(0, n, size=n)
        trees.append(_build(X_tr[idx], y[idx], 0))

    def _ensemble_predict(X):
        all_p = np.array([[_pred(X[i], t) for t in trees] for i in range(len(X))])
        return np.array([np.bincount(row.astype(int)).argmax() for row in all_p])

    predictions = _ensemble_predict(X_te)
    train_preds = _ensemble_predict(X_tr)
    train_acc = float(np.mean(train_preds == y))

    return DescriptiveResult(
        name="rf_classify_bio",
        value=train_acc,
        extra={
            "predictions": predictions,
            "train_accuracy": train_acc,
            "n_trees": n_trees,
            "max_features": max_features,
        },
    )


rfcls = rf_classify_bio


def cheatsheet() -> str:
    return "rf_classify_bio({}) -> Random forest classifier (pure NumPy)."
