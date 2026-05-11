# morie.fn — function file (hadesllm/morie)
"""Parzen window (kernel density) classifier."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "I find your lack of faith disturbing."


def parzen_classify(X_train, y_train, X_test, h=1.0, **kwargs) -> DescriptiveResult:
    """Parzen window (kernel density estimation) classifier.

    Classifies via maximum class-conditional density using Gaussian kernels.

    .. math::

        \\hat{f}_k(\\mathbf{x}) = \\frac{1}{n_k h^p}
        \\sum_{i \\in C_k} K\\!\\left(\\frac{\\mathbf{x} - \\mathbf{x}_i}{h}\\right)

    Parameters
    ----------
    X_train : array-like of shape (n_train, p)
    y_train : array-like of shape (n_train,)
    X_test : array-like of shape (n_test, p)
    h : float
        Bandwidth (default 1.0).

    Returns
    -------
    DescriptiveResult

    References
    ----------
    Parzen, E. (1962). On estimation of a probability density function
        and mode. *Ann. Math. Statist.*, 33(3), 1065--1076.
    """
    X_tr = np.asarray(X_train, dtype=float)
    X_te = np.asarray(X_test, dtype=float)
    y = np.asarray(y_train).ravel()
    classes = np.unique(y)
    p = X_tr.shape[1]

    def _gaussian_kernel(dists_sq):
        return np.exp(-0.5 * dists_sq) / ((2 * np.pi) ** (p / 2))

    predictions = np.empty(len(X_te), dtype=classes.dtype)
    densities = np.zeros((len(X_te), len(classes)))

    for k, c in enumerate(classes):
        X_c = X_tr[y == c]
        n_c = len(X_c)
        for i in range(len(X_te)):
            diff = (X_te[i] - X_c) / h
            dist_sq = np.sum(diff**2, axis=1)
            densities[i, k] = np.sum(_gaussian_kernel(dist_sq)) / (n_c * h**p)

    for i in range(len(X_te)):
        predictions[i] = classes[np.argmax(densities[i])]

    train_densities = np.zeros((len(X_tr), len(classes)))
    for k, c in enumerate(classes):
        X_c = X_tr[y == c]
        n_c = len(X_c)
        for i in range(len(X_tr)):
            diff = (X_tr[i] - X_c) / h
            dist_sq = np.sum(diff**2, axis=1)
            train_densities[i, k] = np.sum(_gaussian_kernel(dist_sq)) / (n_c * h**p)
    train_preds = classes[np.argmax(train_densities, axis=1)]
    train_acc = float(np.mean(train_preds == y))

    return DescriptiveResult(
        name="parzen_classify",
        value=train_acc,
        extra={
            "predictions": predictions,
            "densities": densities,
            "train_accuracy": train_acc,
            "bandwidth": h,
        },
    )


parzn = parzen_classify


def cheatsheet() -> str:
    return "parzen_classify({}) -> Parzen window (kernel density) classifier."
