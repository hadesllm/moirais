"""Mini-batch gradient descent for linear regression."""
import numpy as np

from ._richresult import RichResult

__all__ = ["mini_batch_gradient"]


def mini_batch_gradient(x, y, *, lr=0.01, n_epochs=200, batch_size=32, seed=0):
    """Mini-batch stochastic gradient descent for OLS.

    theta -= lr * (1/B) sum_{i in batch} 2 x_i (x_i' theta - y_i),
    cycling through random batches each epoch.

    Parameters
    ----------
    x : array-like, shape (n,) or (n, p).
    y : array-like, shape (n,).
    lr : float
        Learning rate.
    n_epochs : int
        Number of epochs (full passes through the data).
    batch_size : int
        Mini-batch size.
    seed : int
        RNG seed for batch shuffling (reproducibility).

    Returns
    -------
    RichResult with payload: estimate (theta with intercept), reference_ols,
    n_epochs, batch_size, loss, n, method.
    """
    from sklearn.linear_model import LinearRegression

    X = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    X1 = np.column_stack([np.ones(n), X])
    theta = np.zeros(p + 1)
    rng = np.random.default_rng(seed)
    for _ in range(n_epochs):
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            j = idx[start:start + batch_size]
            xb, yb = X1[j], y[j]
            grad = (2.0 / len(j)) * xb.T @ (xb @ theta - yb)
            theta = theta - lr * grad
    loss = float(np.mean((X1 @ theta - y) ** 2))
    ref = LinearRegression().fit(X, y)
    ref_coef = np.concatenate([[ref.intercept_], ref.coef_])
    return RichResult(payload={
        "estimate": theta.tolist(),
        "reference_ols": ref_coef.tolist(),
        "n_epochs": int(n_epochs),
        "batch_size": int(batch_size),
        "loss": loss,
        "n": int(n),
        "method": "Mini-batch SGD (linear regression)",
    })


def cheatsheet():
    return "mbgrd: mini-batch SGD"


# CANONICAL TEST
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    y = X @ np.array([1.0, -0.5, 2.0]) + 0.25 + rng.normal(scale=0.1, size=500)
    r = mini_batch_gradient(X, y, lr=0.02, n_epochs=100, batch_size=32, seed=0)
    print("theta:", r.estimate)
    print("reference OLS:", r.reference_ols)
    print("final loss:", r.loss)
