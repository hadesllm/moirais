# morie.fn — function file (hadesllm/morie)
"""Simple MLP (1 hidden layer, numpy)."""

import numpy as np

from ._containers import DescriptiveResult


def mlp_simple(
    X: np.ndarray, y: np.ndarray, hidden: int = 32, n_iter: int = 500, lr: float = 0.01, seed: int = 42
) -> DescriptiveResult:
    """
    Simple multi-layer perceptron with one hidden layer (pure numpy).

    Uses ReLU activation in hidden layer and MSE loss for regression.

    :param X: (n, p) feature matrix.
    :param y: (n,) target.
    :param hidden: Number of hidden units.
    :param n_iter: Training iterations (full-batch gradient descent).
    :param lr: Learning rate.
    :param seed: Random seed.
    :return: DescriptiveResult with predictions and training loss.

    References
    ----------
    Rumelhart DE, Hinton GE, Williams RJ (1986). Learning
    representations by back-propagating errors. Nature, 323, 533-536.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n, p = X.shape
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((p, hidden)) * np.sqrt(2.0 / p)
    b1 = np.zeros(hidden)
    W2 = rng.standard_normal((hidden, 1)) * np.sqrt(2.0 / hidden)
    b2 = np.zeros(1)
    losses = []
    for _ in range(n_iter):
        z1 = X @ W1 + b1
        a1 = np.maximum(z1, 0)
        out = (a1 @ W2 + b2).ravel()
        loss = np.mean((out - y) ** 2)
        losses.append(float(loss))
        d_out = 2 * (out - y) / n
        dW2 = a1.T @ d_out.reshape(-1, 1)
        db2 = d_out.sum(keepdims=True)
        d_a1 = d_out.reshape(-1, 1) @ W2.T
        d_z1 = d_a1 * (z1 > 0)
        dW1 = X.T @ d_z1
        db1 = d_z1.sum(axis=0)
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1
    preds = np.maximum(X @ W1 + b1, 0) @ W2 + b2
    preds = preds.ravel()
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return DescriptiveResult(
        name="mlp_simple",
        value=r2,
        extra={
            "predictions": preds,
            "r_squared": r2,
            "final_loss": losses[-1] if losses else 0.0,
            "hidden": hidden,
            "n_iter": n_iter,
            "n": n,
        },
    )


mlp = mlp_simple


def cheatsheet() -> str:
    return "mlp_simple({}) -> Simple MLP (1 hidden layer, numpy)."
