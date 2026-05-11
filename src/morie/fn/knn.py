# morie.fn — function file (hadesllm/morie)
"""k-Nearest Neighbors classifier (pure NumPy)."""

from __future__ import annotations

from typing import Any, Union

import numpy as np
from ._richresult import RichResult


def knn_classify(
    X_train: Union[np.ndarray, Any],
    y_train: Union[np.ndarray, Any],
    X_test: Union[np.ndarray, Any],
    *,
    k: int = 5,
) -> dict[str, Any]:
    """k-Nearest Neighbors classifier using Euclidean distance.

    Pure NumPy implementation — no sklearn required.

    Parameters
    ----------
    X_train : array-like of shape (n_train, p)
        Training feature matrix.
    y_train : array-like of shape (n_train,)
        Training labels.
    X_test : array-like of shape (n_test, p)
        Test feature matrix.
    k : int
        Number of neighbours (default 5).

    Returns
    -------
    dict
        predictions (n_test,), distances (n_test, k).

    References
    ----------
    Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification.
        *IEEE Transactions on Information Theory*, 13(1), 21-27.
    """
    X_tr = np.asarray(X_train, dtype=float)
    y_tr = np.asarray(y_train).ravel()
    X_te = np.asarray(X_test, dtype=float)

    if X_tr.shape[0] != y_tr.shape[0]:
        raise ValueError("X_train and y_train must have same number of rows.")
    if k < 1:
        raise ValueError("k must be >= 1.")
    if k > X_tr.shape[0]:
        raise ValueError(f"k={k} exceeds training set size {X_tr.shape[0]}.")

    n_test = X_te.shape[0]
    predictions = np.empty(n_test, dtype=y_tr.dtype)
    knn_dists = np.empty((n_test, k))

    for i in range(n_test):
        dists = np.sqrt(np.sum((X_tr - X_te[i]) ** 2, axis=1))
        idx = np.argpartition(dists, k)[:k]
        idx = idx[np.argsort(dists[idx])]
        knn_dists[i] = dists[idx]
        # Majority vote
        neighbours = y_tr[idx]
        labels, counts = np.unique(neighbours, return_counts=True)
        predictions[i] = labels[np.argmax(counts)]

    return RichResult(payload={"predictions": predictions, "distances": knn_dists})


knn = knn_classify


def cheatsheet() -> str:
    return "knn_classify({}) -> k-Nearest Neighbors classifier (pure NumPy)."
