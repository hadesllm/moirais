# morie.fn — function file (hadesllm/morie)
"""Learning curve: train/test error vs. training set size."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Union

import numpy as np


def learning_curve(
    model_fn: Callable,
    X: Union[np.ndarray, Any],
    y: Union[np.ndarray, Any],
    *,
    fractions: list[float] | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compute learning curve: train/test accuracy at increasing data sizes.

    Parameters
    ----------
    model_fn : callable
        A function ``model_fn(X_train, y_train, X_test) -> predictions``
        that trains on (X_train, y_train) and returns predictions for X_test.
    X : array-like of shape (n, p)
        Feature matrix.
    y : array-like of shape (n,)
        Labels.
    fractions : list of float or None
        Training set size fractions (default [0.1, 0.3, 0.5, 0.7, 1.0]).
    n_splits : int
        Number of random train/test splits per fraction (default 5).
    random_state : int
        Random seed.

    Returns
    -------
    dict
        fractions, train_scores (mean accuracy), test_scores (mean accuracy).

    References
    ----------
    Perlich, C. (2010). Learning curves in machine learning. *Encyclopedia of
        Machine Learning*, 577-580. Springer.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")
    if fractions is None:
        fractions = [0.1, 0.3, 0.5, 0.7, 1.0]

    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    test_size = max(1, int(n * 0.2))

    train_scores = []
    test_scores = []

    for frac in fractions:
        train_accs = []
        test_accs = []
        for _ in range(n_splits):
            idx = rng.permutation(n)
            test_idx = idx[:test_size]
            avail = idx[test_size:]
            train_n = max(1, int(len(avail) * frac))
            train_idx = avail[:train_n]

            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx], y[test_idx]

            preds_train = model_fn(X_tr, y_tr, X_tr)
            preds_test = model_fn(X_tr, y_tr, X_te)

            train_accs.append(float(np.mean(np.asarray(preds_train) == y_tr)))
            test_accs.append(float(np.mean(np.asarray(preds_test) == y_te)))

        train_scores.append(float(np.mean(train_accs)))
        test_scores.append(float(np.mean(test_accs)))

    return {
        "fractions": fractions,
        "train_scores": train_scores,
        "test_scores": test_scores,
    }


learn = learning_curve


def cheatsheet() -> str:
    return "learning_curve({}) -> Learning curve: train/test error vs. training set size."
