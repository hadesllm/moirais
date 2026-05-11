"""Stacking ensemble classifier."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Power! Unlimited power!"


def stacking_classify(X_train, y_train, X_test, base_clfs=None, cv=3, **kwargs) -> DescriptiveResult:
    """Stacking classifier using cross-validated base learner predictions.

    Default base classifiers: nearest centroid and decision stump.
    Meta-learner: logistic regression (gradient descent).

    Parameters
    ----------
    X_train : array-like of shape (n_train, p)
    y_train : array-like of shape (n_train,)
    X_test : array-like of shape (n_test, p)
    base_clfs : list of callables or None
        Each takes (X_tr, y_tr, X_te) and returns predictions. Default uses
        nearest centroid and a 1-depth tree.
    cv : int
        CV folds for stacking (default 3).

    Returns
    -------
    DescriptiveResult

    References
    ----------
    Wolpert, D.H. (1992). Stacked generalization. *Neural Networks*,
        5(2), 241--259.
    """
    X_tr = np.asarray(X_train, dtype=float)
    X_te = np.asarray(X_test, dtype=float)
    y = np.asarray(y_train).ravel()
    n = len(X_tr)
    classes = np.unique(y)
    y_bin = np.where(y == classes[1], 1.0, 0.0) if len(classes) == 2 else y.astype(float)

    def _nearest_centroid(Xtr, ytr, Xte):
        cs = np.unique(ytr)
        centroids = np.array([Xtr[ytr == c].mean(axis=0) for c in cs])
        dists = np.array([np.sum((Xte - c) ** 2, axis=1) for c in centroids])
        return cs[np.argmin(dists, axis=0)]

    def _stump(Xtr, ytr, Xte):
        from .dtree import decision_tree

        res = decision_tree(Xtr, ytr, Xte, max_depth=1)
        return res.extra["predictions"]

    if base_clfs is None:
        base_clfs = [_nearest_centroid, _stump]

    n_base = len(base_clfs)
    rng = np.random.default_rng(kwargs.get("seed", 42))
    indices = rng.permutation(n)
    fold_size = n // cv

    meta_train = np.zeros((n, n_base))
    meta_test = np.zeros((len(X_te), n_base))

    for b, clf in enumerate(base_clfs):
        test_preds = np.zeros((cv, len(X_te)))
        for fold in range(cv):
            val_idx = indices[fold * fold_size : (fold + 1) * fold_size]
            tr_idx = np.concatenate([indices[: fold * fold_size], indices[(fold + 1) * fold_size :]])
            preds_val = clf(X_tr[tr_idx], y[tr_idx], X_tr[val_idx])
            meta_train[val_idx, b] = preds_val.astype(float)
            test_preds[fold] = clf(X_tr[tr_idx], y[tr_idx], X_te).astype(float)
        meta_test[:, b] = test_preds.mean(axis=0)

    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    w = np.zeros(n_base)
    bias = 0.0
    lr = 0.1
    for _ in range(300):
        z = meta_train @ w + bias
        p = _sigmoid(z)
        grad_w = meta_train.T @ (p - y_bin) / n
        grad_b = (p - y_bin).mean()
        w -= lr * grad_w
        bias -= lr * grad_b

    test_scores = _sigmoid(meta_test @ w + bias)
    predictions = np.where(test_scores >= 0.5, classes[-1], classes[0])
    train_scores = _sigmoid(meta_train @ w + bias)
    train_acc = float(np.mean((train_scores >= 0.5) == y_bin))

    return DescriptiveResult(
        name="stacking_classify",
        value=train_acc,
        extra={
            "predictions": predictions,
            "meta_scores": test_scores,
            "meta_weights": w,
            "train_accuracy": train_acc,
            "n_base_classifiers": n_base,
        },
    )


stckc = stacking_classify


def cheatsheet() -> str:
    return "stacking_classify({}) -> Stacking ensemble classifier."
