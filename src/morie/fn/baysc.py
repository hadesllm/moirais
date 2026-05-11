# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian classifier (Gaussian generative model)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def baysc_fn(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> DescriptiveResult:
    """Classify test samples using Bayes' rule with Gaussian class-conditionals.

    :param X_train: Training features (samples x features).
    :param y_train: Training labels.
    :param X_test: Test features (samples x features).
    :return: DescriptiveResult with prediction count and predictions/posteriors.
    """
    from morie._classify import bayes_classifier

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=float)
    preds, posteriors = bayes_classifier(X_train, y_train, X_test)
    acc = float(np.mean(preds == y_train[: len(preds)])) if len(preds) <= len(y_train) else 0.0
    return DescriptiveResult(
        name="bayes_classifier",
        value=len(preds),
        extra={"predictions": preds, "posteriors": posteriors},
    )


baysc = baysc_fn


def cheatsheet() -> str:
    return "baysc_fn({}) -> Bayesian classifier (Gaussian generative model)."
