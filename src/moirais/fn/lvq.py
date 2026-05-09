# moirais.fn — function file (hadesllm/moirais)
"""Learning vector quantization classifier."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "You were the chosen one!"


def learning_vq(X_train, y_train, X_test, n_proto=10, lr=0.01, n_iter=200, **kwargs) -> DescriptiveResult:
    """Learning vector quantization (LVQ1) classifier.

    Prototypes are initialised from random training samples per class
    and updated via competitive learning.

    Parameters
    ----------
    X_train : array-like of shape (n_train, p)
    y_train : array-like of shape (n_train,)
    X_test : array-like of shape (n_test, p)
    n_proto : int
        Number of prototypes per class (default 10).
    lr : float
        Learning rate (default 0.01).
    n_iter : int
        Epochs (default 200).

    Returns
    -------
    DescriptiveResult

    References
    ----------
    Kohonen, T. (1990). The self-organizing map. *Proc. IEEE*,
        78(9), 1464--1480.
    """
    X_tr = np.asarray(X_train, dtype=float)
    X_te = np.asarray(X_test, dtype=float)
    y = np.asarray(y_train).ravel()
    classes = np.unique(y)
    rng = np.random.default_rng(kwargs.get("seed", 42))

    protos = []
    proto_labels = []
    for c in classes:
        idx = np.where(y == c)[0]
        chosen = rng.choice(idx, size=min(n_proto, len(idx)), replace=len(idx) < n_proto)
        protos.append(X_tr[chosen].copy())
        proto_labels.extend([c] * len(chosen))
    protos = np.vstack(protos)
    proto_labels = np.array(proto_labels)

    n = len(X_tr)
    for _ in range(n_iter):
        order = rng.permutation(n)
        for i in order:
            dists = np.sum((protos - X_tr[i]) ** 2, axis=1)
            winner = np.argmin(dists)
            if proto_labels[winner] == y[i]:
                protos[winner] += lr * (X_tr[i] - protos[winner])
            else:
                protos[winner] -= lr * (X_tr[i] - protos[winner])

    def _predict(X):
        preds = np.empty(len(X), dtype=proto_labels.dtype)
        for i in range(len(X)):
            dists = np.sum((protos - X[i]) ** 2, axis=1)
            preds[i] = proto_labels[np.argmin(dists)]
        return preds

    predictions = _predict(X_te)
    train_acc = float(np.mean(_predict(X_tr) == y))

    return DescriptiveResult(
        name="learning_vq",
        value=train_acc,
        extra={
            "predictions": predictions,
            "prototypes": protos,
            "prototype_labels": proto_labels,
            "train_accuracy": train_acc,
        },
    )


lvq = learning_vq


def cheatsheet() -> str:
    return "learning_vq({}) -> Learning vector quantization classifier."
