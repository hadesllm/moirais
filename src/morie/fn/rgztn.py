"""Ridge / LASSO / ElasticNet regularization path."""
import numpy as np

from ._richresult import RichResult

__all__ = ["regularization_path"]


def regularization_path(x, y, *, penalty="ridge", alphas=None, l1_ratio=0.5):
    """Coefficient path beta(lambda) across a grid of penalties.

    Wraps sklearn.linear_model.Ridge / Lasso / ElasticNet.  Returns the
    coefficient matrix (one row per lambda) plus the alpha grid used.

    Parameters
    ----------
    x : array-like (n, p).
    y : array-like (n,).
    penalty : {"ridge", "lasso", "elasticnet"}
        L2, L1, or convex combination.
    alphas : array-like or None
        Regularization strengths.  Defaults to logspace(-3, 2, 50).
    l1_ratio : float
        Only used for elasticnet (0 -> ridge, 1 -> lasso).

    Returns
    -------
    RichResult with payload: estimate (coefficients of last/strongest model),
    coef_path (shape (len(alphas), p+1) including intercept), alphas, penalty,
    n, method.
    """
    from sklearn.linear_model import Ridge, Lasso, ElasticNet

    X = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    if alphas is None:
        alphas = np.logspace(-3, 2, 50)
    alphas = np.asarray(alphas, dtype=float)

    coef_path = np.zeros((len(alphas), p + 1))
    for i, a in enumerate(alphas):
        if penalty == "ridge":
            m = Ridge(alpha=a, fit_intercept=True)
        elif penalty == "lasso":
            m = Lasso(alpha=a, fit_intercept=True, max_iter=20000)
        elif penalty == "elasticnet":
            m = ElasticNet(alpha=a, l1_ratio=l1_ratio, fit_intercept=True, max_iter=20000)
        else:
            raise ValueError(f"unknown penalty: {penalty}")
        m.fit(X, y)
        coef_path[i, 0] = m.intercept_
        coef_path[i, 1:] = m.coef_
    return RichResult(payload={
        "estimate": coef_path[-1].tolist(),
        "coef_path": coef_path.tolist(),
        "alphas": alphas.tolist(),
        "penalty": penalty,
        "l1_ratio": float(l1_ratio) if penalty == "elasticnet" else None,
        "n": int(n),
        "method": f"Regularization path ({penalty})",
    })


def cheatsheet():
    return "rgztn: ridge/lasso/elasticnet regularization path"


# CANONICAL TEST
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = X @ np.array([1.0, 0.0, -0.5, 0.0, 2.0]) + rng.normal(scale=0.1, size=200)
    r = regularization_path(X, y, penalty="ridge", alphas=[0.01, 0.1, 1.0, 10.0])
    print("alphas:", r.alphas)
    print("coef path:", r.coef_path)
    print("strongest-penalty estimate:", r.estimate)
