# moirais.fn — function file (hadesllm/moirais)
"""Gaussian Naive Bayes (pure numpy). 'Out of chaos, comes order. — Friedrich Nietzsche'

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def naive_bayes_fit(X: np.ndarray, y: np.ndarray) -> dict:
    """Fit a Gaussian Naive Bayes model.

    Parameters
    ----------
    X : ndarray, shape (n, p)
    y : ndarray, shape (n,)
        Integer class labels.

    Returns
    -------
    dict
        Model containing class priors, means, and variances.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes = np.unique(y)
    model = {"classes": classes, "priors": {}, "means": {}, "vars": {}}
    for c in classes:
        mask = y == c
        model["priors"][c] = float(np.mean(mask))
        model["means"][c] = np.mean(X[mask], axis=0)
        model["vars"][c] = np.var(X[mask], axis=0) + 1e-9
    return model


def naive_bayes_predict(model: dict, X_new: np.ndarray) -> np.ndarray:
    """Predict class labels.

    Parameters
    ----------
    model : dict
        From ``naive_bayes_fit``.
    X_new : ndarray

    Returns
    -------
    ndarray
    """
    X_new = np.asarray(X_new, dtype=float)
    classes = model["classes"]
    log_probs = np.zeros((len(X_new), len(classes)))
    for i, c in enumerate(classes):
        prior = np.log(model["priors"][c])
        mean = model["means"][c]
        var = model["vars"][c]
        log_lik = -0.5 * np.sum(np.log(2 * np.pi * var) + (X_new - mean) ** 2 / var, axis=1)
        log_probs[:, i] = prior + log_lik
    return classes[np.argmax(log_probs, axis=1)]


def naive_bayes(
    X: np.ndarray,
    y: np.ndarray,
    X_pred: np.ndarray | None = None,
) -> DescriptiveResult:
    """Fit and optionally predict with Gaussian Naive Bayes.

    Parameters
    ----------
    X : ndarray
    y : ndarray
    X_pred : ndarray or None

    Returns
    -------
    DescriptiveResult
    """
    model = naive_bayes_fit(X, y)
    if X_pred is None:
        X_pred = X
    preds = naive_bayes_predict(model, X_pred)
    accuracy = float(np.mean(preds == y[: len(preds)])) if len(preds) == len(y) else None
    return DescriptiveResult(
        name="Gaussian Naive Bayes",
        value=accuracy,
        extra={
            "predictions": preds,
            "classes": model["classes"],
            "n_train": len(y),
            "n_pred": len(preds),
        },
    )


nb_ = naive_bayes


def cheatsheet() -> str:
    return "naive_bayes_fit({}) -> Gaussian Naive Bayes (pure numpy). 'These blast points... to"
