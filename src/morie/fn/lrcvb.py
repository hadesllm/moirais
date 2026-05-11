# morie.fn — function file (hadesllm/morie)
"""Learning curve analysis."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "In a dark place we find ourselves, and a little more knowledge lights our way."


def learning_curve_bio(X, y, train_sizes=None, cv=5, **kwargs) -> DescriptiveResult:
    """Learning curve analysis via cross-validated accuracy at varying train sizes.

    Uses a simple nearest-centroid classifier as the default learner.

    Parameters
    ----------
    X : array-like of shape (n, p)
    y : array-like of shape (n,)
    train_sizes : array-like or None
        Fractions of training data to use. Default: [0.1, 0.3, 0.5, 0.7, 0.9].
    cv : int
        Number of folds (default 5).

    Returns
    -------
    DescriptiveResult
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()
    n = len(X)
    if train_sizes is None:
        train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9]
    train_sizes = np.asarray(train_sizes)

    rng = np.random.default_rng(kwargs.get("seed", 42))
    indices = rng.permutation(n)
    fold_size = n // cv

    results_train = []
    results_val = []

    for frac in train_sizes:
        train_accs = []
        val_accs = []
        for fold in range(cv):
            val_idx = indices[fold * fold_size : (fold + 1) * fold_size]
            train_idx = np.concatenate([indices[: fold * fold_size], indices[(fold + 1) * fold_size :]])
            m = max(1, int(frac * len(train_idx)))
            tr_idx = train_idx[:m]

            classes = np.unique(y[tr_idx])
            centroids = np.array([X[tr_idx][y[tr_idx] == c].mean(axis=0) for c in classes])

            def _predict(Xp):
                dists = np.array([np.sum((Xp - c) ** 2, axis=1) for c in centroids])
                return classes[np.argmin(dists, axis=0)]

            train_accs.append(float(np.mean(_predict(X[tr_idx]) == y[tr_idx])))
            val_accs.append(float(np.mean(_predict(X[val_idx]) == y[val_idx])))

        results_train.append(np.mean(train_accs))
        results_val.append(np.mean(val_accs))

    return DescriptiveResult(
        name="learning_curve_bio",
        value=float(results_val[-1]),
        extra={
            "train_sizes": train_sizes,
            "train_scores": np.array(results_train),
            "val_scores": np.array(results_val),
            "cv": cv,
        },
    )


lrcvb = learning_curve_bio


def cheatsheet() -> str:
    return "learning_curve_bio({}) -> Learning curve analysis."
