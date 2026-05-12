# morie.fn -- function file (hadesllm/morie)
"""Simple neural network (1 hidden layer, pure NumPy)."""

from __future__ import annotations

from typing import Any, Union

import numpy as np
from ._richresult import RichResult


def nn_classify(
    X: Union[np.ndarray, Any],
    y: Union[np.ndarray, Any],
    *,
    hidden_size: int = 32,
    epochs: int = 100,
    lr: float = 0.01,
    random_state: int = 42,
) -> dict[str, Any]:
    """Single hidden-layer neural network for binary classification.

    Architecture: input -> Dense(hidden_size, sigmoid) -> Dense(1, sigmoid).
    Trained with binary cross-entropy via gradient descent.
    Pure NumPy -- no external ML library required.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Feature matrix.
    y : array-like of shape (n,)
        Binary labels (0/1).
    hidden_size : int
        Number of hidden units (default 32).
    epochs : int
        Training epochs (default 100).
    lr : float
        Learning rate (default 0.01).
    random_state : int
        Random seed.

    Returns
    -------
    dict
        predictions (n,), loss_history (list of float).

    References
    ----------
    Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning
        representations by back-propagating errors. *Nature*, 323, 533-536.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")

    rng = np.random.default_rng(random_state)
    n, p = X.shape

    # Xavier initialisation
    W1 = rng.standard_normal((p, hidden_size)) * np.sqrt(2.0 / p)
    b1 = np.zeros(hidden_size)
    W2 = rng.standard_normal((hidden_size, 1)) * np.sqrt(2.0 / hidden_size)
    b2 = np.zeros(1)

    loss_history: list[float] = []

    for _ in range(epochs):
        # Forward
        z1 = X @ W1 + b1
        a1 = _sigmoid(z1)
        z2 = a1 @ W2 + b2
        a2 = _sigmoid(z2).ravel()

        # Loss (binary cross-entropy)
        eps = 1e-12
        loss = -np.mean(y * np.log(a2 + eps) + (1 - y) * np.log(1 - a2 + eps))
        loss_history.append(float(loss))

        # Backward
        dz2 = (a2 - y).reshape(-1, 1)  # (n, 1)
        dW2 = (a1.T @ dz2) / n
        db2 = np.mean(dz2, axis=0)

        da1 = dz2 @ W2.T  # (n, hidden)
        dz1 = da1 * a1 * (1 - a1)
        dW1 = (X.T @ dz1) / n
        db1 = np.mean(dz1, axis=0)

        # Update
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

    # Final predictions
    z1 = X @ W1 + b1
    a1 = _sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = _sigmoid(z2).ravel()
    preds = (a2 >= 0.5).astype(float)

    return RichResult(payload={"predictions": preds, "loss_history": loss_history})


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


nn_ = nn_classify


def cheatsheet() -> str:
    return "nn_classify({}) -> Simple neural network (1 hidden layer, pure NumPy)."
