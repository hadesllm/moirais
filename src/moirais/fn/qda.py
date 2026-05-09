# moirais.fn — function file (hadesllm/moirais)
"""Quadratic Discriminant Analysis classifier."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def qda_fn(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> DescriptiveResult:
    """Classify test samples using Quadratic Discriminant Analysis.

    :param X_train: Training features (samples x features).
    :param y_train: Training labels.
    :param X_test: Test features (samples x features).
    :return: DescriptiveResult with prediction count and predictions.
    """
    from moirais._classify import qda_classify

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=float)
    preds = qda_classify(X_train, y_train, X_test)
    return DescriptiveResult(
        name="qda",
        value=len(preds),
        extra={"predictions": preds},
    )


qda = qda_fn


def cheatsheet() -> str:
    return "qda_fn({}) -> Quadratic Discriminant Analysis classifier."
