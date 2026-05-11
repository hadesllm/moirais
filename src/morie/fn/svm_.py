"""Support Vector Machine classifier wrapper."""

from __future__ import annotations

from typing import Any, Union

import numpy as np
from ._richresult import RichResult


def svm_classify(
    X: Union[np.ndarray, Any],
    y: Union[np.ndarray, Any],
    *,
    kernel: str = "rbf",
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> dict[str, Any]:
    """SVM classifier with sklearn fallback to pure-NumPy linear SGD.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Feature matrix.
    y : array-like of shape (n,)
        Binary labels (0/1 or -1/+1).
    kernel : str
        Kernel type for sklearn SVC (default ``"rbf"``).
    C : float
        Regularisation parameter (default 1.0).
    max_iter : int
        Maximum iterations for SGD fallback (default 1000).
    random_state : int
        Random seed.

    Returns
    -------
    dict
        predictions, support_vectors (indices), accuracy.

    References
    ----------
    Cortes, C., & Vapnik, V. (1995). Support-vector networks.
        *Machine Learning*, 20(3), 273-297. doi:10.1007/BF00994018
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")

    try:
        from sklearn.svm import SVC

        model = SVC(kernel=kernel, C=C, random_state=random_state)
        model.fit(X, y)
        preds = model.predict(X)
        acc = float(np.mean(preds == y))
        sv_idx = model.support_
        return RichResult(payload={"predictions": preds, "support_vectors": sv_idx, "accuracy": acc})
    except ImportError:
        pass

    # Pure-numpy linear SVM via SGD (hinge loss)
    labels = y.copy()
    unique_labels = np.unique(labels)
    if set(unique_labels) == {0, 1}:
        labels = 2 * labels - 1  # convert to -1/+1

    rng = np.random.default_rng(random_state)
    n, p = X.shape
    w = rng.standard_normal(p) * 0.01
    b = 0.0
    lr = 0.01

    for epoch in range(max_iter):
        idx = rng.permutation(n)
        for i in idx:
            margin = labels[i] * (X[i] @ w + b)
            if margin < 1:
                w += lr * (labels[i] * X[i] - (1.0 / C) * w / n)
                b += lr * labels[i]
            else:
                w -= lr * (1.0 / C) * w / n

    scores = X @ w + b
    preds_raw = np.sign(scores)
    # Convert back to original label space
    if set(unique_labels) == {0, 1}:
        preds = ((preds_raw + 1) / 2).astype(float)
    else:
        preds = preds_raw

    # Approximate support vectors: points close to margin
    margins = np.abs(X @ w + b)
    sv_idx = np.where(margins < 1.0)[0]

    acc = float(np.mean(preds == y))
    return RichResult(payload={"predictions": preds, "support_vectors": sv_idx, "accuracy": acc})


svm_ = svm_classify


def cheatsheet() -> str:
    return "svm_classify({}) -> Support Vector Machine classifier wrapper."
