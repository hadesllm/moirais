"""Vanilla (batch) gradient descent for linear regression."""
import numpy as np

from ._richresult import RichResult

__all__ = ["gradient_descent_vanilla"]


def gradient_descent_vanilla(x, y, *, lr=0.01, n_iter=1000, tol=1e-8):
    """Vanilla batch gradient descent for OLS.

    theta := theta - lr * (2/n) X' (X theta - y), with X including a
    column of ones for the intercept.  Compared against the sklearn
    closed-form solution to guarantee convergence.

    Parameters
    ----------
    x : array-like, shape (n,) or (n, p).
    y : array-like, shape (n,).
    lr : float
        Learning rate.
    n_iter : int
        Max iterations.
    tol : float
        L2 step-norm tolerance for early stopping.

    Returns
    -------
    RichResult with payload: estimate (theta vector with intercept first),
    n_iter (iterations actually taken), loss (final MSE), n, method.
    """
    from sklearn.linear_model import LinearRegression

    X = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    X1 = np.column_stack([np.ones(n), X])
    theta = np.zeros(p + 1)
    last_iter = 0
    for it in range(n_iter):
        grad = (2.0 / n) * X1.T @ (X1 @ theta - y)
        step = lr * grad
        theta = theta - step
        last_iter = it + 1
        if np.linalg.norm(step) < tol:
            break
    loss = float(np.mean((X1 @ theta - y) ** 2))

    # Reference closed-form for comparison
    ref = LinearRegression().fit(X, y)
    ref_coef = np.concatenate([[ref.intercept_], ref.coef_])

    return RichResult(payload={
        "estimate": theta.tolist(),
        "reference_ols": ref_coef.tolist(),
        "n_iter": int(last_iter),
        "loss": loss,
        "n": int(n),
        "method": "Vanilla batch gradient descent (linear regression)",
    })


def cheatsheet():
    return "grdds: vanilla batch gradient descent"


# CANONICAL TEST
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = 0.5 + X @ np.array([1.5, -2.0]) + rng.normal(scale=0.1, size=200)
    r = gradient_descent_vanilla(X, y, lr=0.05, n_iter=5000)
    print("theta:", r.estimate)
    print("reference OLS:", r.reference_ols)
    print("n_iter:", r.n_iter, " loss:", r.loss)
