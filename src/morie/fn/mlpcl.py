# morie.fn -- function file (hadesllm/morie)
"""Multilayer perceptron classifier (pure NumPy)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Luminous beings are we, not this crude matter."


def mlp_classify(X_train, y_train, X_test, hidden=(64, 32), lr=0.01, n_iter=500, **kwargs) -> DescriptiveResult:
    """Feedforward neural network for binary classification.

    Uses ReLU activations in hidden layers and sigmoid output.
    Trained with mini-batch gradient descent and binary cross-entropy loss.

    Parameters
    ----------
    X_train : array-like of shape (n_train, p)
    y_train : array-like of shape (n_train,)
    X_test : array-like of shape (n_test, p)
    hidden : tuple of int
        Hidden layer sizes (default (64, 32)).
    lr : float
        Learning rate (default 0.01).
    n_iter : int
        Training epochs (default 500).

    Returns
    -------
    DescriptiveResult

    References
    ----------
    Rumelhart, D.E., Hinton, G.E. & Williams, R.J. (1986). Learning
        representations by back-propagating errors. *Nature*, 323, 533--536.
    """
    X_tr = np.asarray(X_train, dtype=float)
    X_te = np.asarray(X_test, dtype=float)
    y = np.asarray(y_train).ravel()
    classes = np.unique(y)
    y_bin = np.where(y == classes[1], 1.0, 0.0).reshape(-1, 1)

    rng = np.random.default_rng(kwargs.get("seed", 42))
    layers = [X_tr.shape[1]] + list(hidden) + [1]
    weights = []
    biases = []
    for i in range(len(layers) - 1):
        scale = np.sqrt(2.0 / layers[i])
        weights.append(rng.normal(0, scale, (layers[i], layers[i + 1])))
        biases.append(np.zeros((1, layers[i + 1])))

    def _relu(z):
        return np.maximum(z, 0)

    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _forward(X):
        activations = [X]
        for i in range(len(weights) - 1):
            z = activations[-1] @ weights[i] + biases[i]
            activations.append(_relu(z))
        z = activations[-1] @ weights[-1] + biases[-1]
        activations.append(_sigmoid(z))
        return activations

    n = len(X_tr)
    losses = []
    for epoch in range(n_iter):
        acts = _forward(X_tr)
        out = acts[-1]
        out_clip = np.clip(out, 1e-7, 1 - 1e-7)
        loss = -np.mean(y_bin * np.log(out_clip) + (1 - y_bin) * np.log(1 - out_clip))
        losses.append(float(loss))

        delta = out - y_bin
        for i in range(len(weights) - 1, -1, -1):
            dw = acts[i].T @ delta / n
            db = delta.mean(axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ weights[i].T) * (acts[i] > 0).astype(float)
            weights[i] -= lr * dw
            biases[i] -= lr * db

    test_out = _forward(X_te)[-1].ravel()
    predictions = np.where(test_out >= 0.5, classes[1], classes[0])
    train_out = _forward(X_tr)[-1].ravel()
    train_acc = float(np.mean((train_out >= 0.5) == y_bin.ravel()))

    return DescriptiveResult(
        name="mlp_classify",
        value=train_acc,
        extra={
            "predictions": predictions,
            "probabilities": test_out,
            "train_accuracy": train_acc,
            "final_loss": losses[-1] if losses else float("nan"),
            "hidden": hidden,
        },
    )


mlpcl = mlp_classify


def cheatsheet() -> str:
    return "mlp_classify({}) -> Multilayer perceptron classifier (pure NumPy)."
