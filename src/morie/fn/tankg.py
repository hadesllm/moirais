"""He who has a why to live can bear almost any how. -- Friedrich Nietzsche"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def nn_train(
    X: np.ndarray,
    y: np.ndarray,
    *,
    hidden: int = 16,
    lr: float = 0.01,
    epochs: int = 200,
    seed: int | None = 42,
) -> DescriptiveResult:
    """Train a single-hidden-layer neural network (binary classification).

    Implements forward/backward propagation with sigmoid activations and
    binary cross-entropy loss. No external ML library required.

    Parameters
    ----------
    X : ndarray
        Feature matrix (n x p).
    y : ndarray
        Binary labels (n,), values 0 or 1.
    hidden : int
        Number of hidden units.
    lr : float
        Learning rate.
    epochs : int
        Number of training epochs.
    seed : int or None
        Random seed.

    Returns
    -------
    DescriptiveResult
        ``value`` is the final training accuracy; ``extra`` has loss history
        and weight shapes.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    if len(y) != X.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    n, p = X.shape

    rng = np.random.default_rng(seed)

    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    W1 = rng.normal(0, 0.5, (p, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, 0.5, (hidden, 1))
    b2 = np.zeros(1)

    losses = []
    for ep in range(epochs):
        z1 = X @ W1 + b1
        a1 = _sigmoid(z1)
        z2 = a1 @ W2 + b2
        a2 = _sigmoid(z2).ravel()

        eps = 1e-12
        loss = -np.mean(y * np.log(a2 + eps) + (1 - y) * np.log(1 - a2 + eps))
        losses.append(float(loss))

        dz2 = (a2 - y).reshape(-1, 1)
        dW2 = a1.T @ dz2 / n
        db2 = dz2.mean(axis=0)

        da1 = dz2 @ W2.T
        dz1 = da1 * a1 * (1 - a1)
        dW1 = X.T @ dz1 / n
        db1 = dz1.mean(axis=0)

        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

    z1 = X @ W1 + b1
    a1 = _sigmoid(z1)
    z2 = a1 @ W2 + b2
    preds = (_sigmoid(z2).ravel() >= 0.5).astype(float)
    accuracy = float(np.mean(preds == y))

    return DescriptiveResult(
        name="NeuralNet (1-hidden)",
        value=accuracy,
        extra={
            "final_loss": losses[-1] if losses else float("nan"),
            "loss_history": losses[:: max(1, len(losses) // 20)],
            "epochs": epochs,
            "hidden": hidden,
            "n": n,
            "p": p,
        },
    )


tankg = nn_train


def cheatsheet() -> str:
    return "He who has a why to live can bear almost any how. -- Friedrich Nietzsche"
