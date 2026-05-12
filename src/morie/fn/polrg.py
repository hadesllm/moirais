"""Polynomial feature expansion + OLS."""
import numpy as np

from ._richresult import RichResult

__all__ = ["polynomial_regression"]


def polynomial_regression(x, y, *, degree=2):
    """Polynomial regression via PolynomialFeatures + LinearRegression.

    y = beta_0 + beta_1 x + beta_2 x^2 + ... + beta_d x^d (univariate)
    or full multivariate polynomial expansion of degree d.

    Parameters
    ----------
    x : array-like, shape (n,) or (n, p).
    y : array-like, shape (n,).
    degree : int
        Polynomial degree.

    Returns
    -------
    RichResult with payload: estimate (coefficients including intercept),
    se (classical OLS standard errors), feature_names, n, method.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    X = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    Xp = poly.fit_transform(X)
    model = LinearRegression(fit_intercept=True).fit(Xp, y)
    coef = np.concatenate([[model.intercept_], model.coef_])
    yhat = model.predict(Xp)
    resid = y - yhat
    p = Xp.shape[1]
    df = max(n - p - 1, 1)
    sigma2 = float(np.sum(resid**2) / df)
    X1 = np.column_stack([np.ones(n), Xp])
    try:
        cov = sigma2 * np.linalg.inv(X1.T @ X1)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        se = np.full(p + 1, np.nan)
    names = ["(intercept)"] + list(poly.get_feature_names_out([f"x{i}" for i in range(X.shape[1])]))
    return RichResult(payload={
        "estimate": coef.tolist(),
        "se": se.tolist(),
        "feature_names": names,
        "degree": int(degree),
        "n": int(n),
        "method": f"Polynomial regression (degree={degree})",
    })


def cheatsheet():
    return "polrg: polynomial regression (PolynomialFeatures + OLS)"


# CANONICAL TEST
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = np.linspace(-2, 2, 200)
    y = 1.0 - 0.5 * x + 2.0 * x**2 + rng.normal(scale=0.1, size=200)
    r = polynomial_regression(x, y, degree=2)
    # Expected: ~[1.0, -0.5, 2.0]
    print("coef:", r.estimate)
    print("se:", r.se)
    print("features:", r.feature_names)
