# morie.fn -- function file (hadesllm/morie)
"""Ordinary least squares closed-form solution (sklearn-backed)."""
import numpy as np

from ._richresult import RichResult

__all__ = ["linear_regression_ols"]


def linear_regression_ols(x, y):
    """Ordinary least squares closed-form solution.

    Solves beta = (X'X)^{-1} X'y via sklearn.linear_model.LinearRegression
    and returns the coefficient vector together with classical OLS
    standard errors sigma * sqrt(diag((X'X)^{-1})).

    Parameters
    ----------
    x : array-like, shape (n,) or (n, p)
        Design matrix (intercept added internally).
    y : array-like, shape (n,)
        Response.

    Returns
    -------
    RichResult with payload keys: estimate (intercept + slopes),
    se (intercept + slopes), n, method.

    References
    ----------
    Hastie, Tibshirani & Friedman, ESL (2009), Ch 3.
    """
    from sklearn.linear_model import LinearRegression

    X = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape

    model = LinearRegression(fit_intercept=True).fit(X, y)
    coef = np.concatenate([[model.intercept_], model.coef_])

    # Classical OLS SE: sigma^2 = RSS/(n - p - 1); Var(b) = sigma^2 (X1' X1)^-1
    yhat = model.predict(X)
    resid = y - yhat
    df = max(n - p - 1, 1)
    sigma2 = float(np.sum(resid**2) / df)
    X1 = np.column_stack([np.ones(n), X])
    try:
        cov = sigma2 * np.linalg.inv(X1.T @ X1)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        se = np.full(p + 1, np.nan)

    return RichResult(payload={
        "estimate": coef.tolist(),
        "se": se.tolist(),
        "n": int(n),
        "method": "OLS via closed-form normal equations",
    })


def cheatsheet():
    return "linrg: OLS regression (closed-form normal equations)"


# CANONICAL TEST
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    beta_true = np.array([1.5, -2.0])
    y = 0.5 + X @ beta_true + rng.normal(scale=0.1, size=200)
    r = linear_regression_ols(X, y)
    # Expected: intercept ~ 0.5, slopes ~ 1.5, -2.0
    print("estimate:", r.estimate)
    print("se:", r.se)
    print("n:", r.n)
