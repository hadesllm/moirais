# morie.fn -- function file (hadesllm/morie)
"""Minimum distance classifier."""

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The man who moves a mountain begins by carrying away small stones. -- Confucius"


def min_distance_classify(X_train, y_train, X_test, **kwargs) -> DescriptiveResult:
    """
    Minimum (Euclidean) distance to class-centroid classifier.

    Computes the centroid of each class, then assigns each test point
    to the class whose centroid is nearest.

    :param X_train: (n_train, d) training features.
    :param y_train: (n_train,) training labels.
    :param X_test: (n_test, d) test features.
    :return: DescriptiveResult with predictions.

    References
    ----------
    Duda RO, Hart PE, Stork DG (2001). Pattern Classification,
    2nd ed. Wiley, New York.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train).ravel()
    X_test = np.asarray(X_test, dtype=np.float64)
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
    classes = np.unique(y_train)
    centroids = np.array([X_train[y_train == c].mean(axis=0) for c in classes])
    preds = []
    for xt in X_test:
        dists = np.sqrt(np.sum((centroids - xt) ** 2, axis=1))
        preds.append(classes[np.argmin(dists)])
    preds = np.array(preds)
    extra = {"n_classes": len(classes), "n_test": len(X_test)}
    y_test = kwargs.get("y_test")
    if y_test is not None:
        y_test = np.asarray(y_test).ravel()
        extra["accuracy"] = float(np.mean(preds == y_test))
    return DescriptiveResult(
        name="min_distance_classify",
        value=extra.get("accuracy"),
        extra={**extra, "predictions": preds.tolist()},
    )


minnr = min_distance_classify


def cheatsheet() -> str:
    return "min_distance_classify({}) -> Minimum distance classifier."
