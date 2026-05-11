# morie.fn — function file (hadesllm/morie)
"""Logistic regression classifier (gradient descent)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def logcl_fn(X: np.ndarray, y: np.ndarray, lr: float = 0.01, n_iter: int = 1000) -> DescriptiveResult:
    """Train a logistic regression classifier via gradient descent.

    :param X: Training features (samples x features).
    :param y: Binary labels (0/1).
    :param lr: Learning rate (default 0.01).
    :param n_iter: Number of iterations (default 1000).
    :return: DescriptiveResult with accuracy and weights.
    """
    from morie._classify import logistic_classify

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w, accuracy = logistic_classify(X, y, lr=lr, n_iter=n_iter)
    return DescriptiveResult(
        name="logistic",
        value=accuracy,
        extra={"weights": w, "accuracy": accuracy},
    )


logcl = logcl_fn


def cheatsheet() -> str:
    return "logcl_fn({}) -> Logistic regression classifier (gradient descent)."
