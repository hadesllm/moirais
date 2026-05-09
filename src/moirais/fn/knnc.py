# moirais.fn — function file (hadesllm/moirais)
"""k-Nearest Neighbors classifier."""

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Stay on target. -- Gold Five"


def knn_classify(X_train, y_train, X_test, k: int = 5, **kwargs) -> DescriptiveResult:
    """The art of doing mathematics consists in finding that special case which contains all the germs of generality. — David Hilbert"""
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train).ravel()
    X_test = np.asarray(X_test, dtype=np.float64)
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
    if k < 1 or k > len(X_train):
        raise ValueError(f"k must be in [1, {len(X_train)}].")
    preds = []
    for xt in X_test:
        dists = np.sqrt(np.sum((X_train - xt) ** 2, axis=1))
        idx = np.argsort(dists)[:k]
        labels, counts = np.unique(y_train[idx], return_counts=True)
        preds.append(labels[np.argmax(counts)])
    preds = np.array(preds)
    extra = {"k": k, "n_train": len(X_train), "n_test": len(X_test)}
    y_test = kwargs.get("y_test")
    if y_test is not None:
        y_test = np.asarray(y_test).ravel()
        extra["accuracy"] = float(np.mean(preds == y_test))
    return DescriptiveResult(
        name="knn_classify",
        value=extra.get("accuracy"),
        extra={**extra, "predictions": preds.tolist()},
    )


knnc = knn_classify


def cheatsheet() -> str:
    return "knn_classify({}) -> k-Nearest Neighbors classifier."
