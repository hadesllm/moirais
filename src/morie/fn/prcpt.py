# morie.fn — function file (hadesllm/morie)
"""Single-layer perceptron."""

import numpy as np

from ._containers import DescriptiveResult


def perceptron(X: np.ndarray, y: np.ndarray, n_iter: int = 100, lr: float = 0.01) -> DescriptiveResult:
    """
    Single-layer perceptron classifier.

    Updates weights only on misclassified examples. Labels {-1, +1}.

    :param X: (n, p) feature matrix.
    :param y: (n,) labels in {-1, +1}.
    :param n_iter: Number of passes.
    :param lr: Learning rate.
    :return: DescriptiveResult with weights and accuracy.

    References
    ----------
    Rosenblatt F (1958). The perceptron: a probabilistic model for
    information storage and organization in the brain.
    Psychological Review, 65(6), 386-408.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n, p = X.shape
    w = np.zeros(p)
    b = 0.0
    n_updates = 0
    for _ in range(n_iter):
        for i in range(n):
            if y[i] * (X[i] @ w + b) <= 0:
                w += lr * y[i] * X[i]
                b += lr * y[i]
                n_updates += 1
    preds = np.sign(X @ w + b)
    accuracy = float(np.mean(preds == y))
    return DescriptiveResult(
        name="perceptron",
        value=accuracy,
        extra={"weights": w, "bias": b, "predictions": preds, "accuracy": accuracy, "n_updates": n_updates, "n": n},
    )


prcpt = perceptron


def cheatsheet() -> str:
    return "perceptron({}) -> Single-layer perceptron."
