"""Spatial autoregressive lag model (SAR lag, ML)."""
import numpy as np
from scipy import optimize
from ._richresult import RichResult

__all__ = ["spatial_ar_lag"]


def spatial_ar_lag(x, y, w):
    """
    SAR lag model:
        Y = rho W Y + X beta + eps,   eps ~ N(0, sigma2 I).

    Concentrated log-likelihood in rho (Anselin 1988; Schabenberger
    & Gotway 2005, Ch 7):

        ll(rho) = -n/2 log(2 pi sigma2_hat) + log|I - rho W| - n/2
        e0 = M y,  e1 = M W y,   M = I - X (X'X)^{-1} X'
        sigma2_hat(rho) = (e0 - rho e1)' (e0 - rho e1) / n.

    Parameters
    ----------
    x : array-like, shape (n, p) -- design matrix (intercept explicit).
    y : array-like, shape (n,)
    w : array-like, shape (n, n)

    Returns
    -------
    RichResult with payload: estimate (beta), se, rho, sigma2, n, method.
    """
    X = np.asarray(x, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y, dtype=float).ravel()
    W = np.asarray(w, dtype=float)
    n, p = X.shape
    if y.size != n or W.shape != (n, n):
        raise ValueError("shape mismatch among x, y, w")
    I = np.eye(n)
    XtX_inv = np.linalg.inv(X.T @ X)
    M = I - X @ XtX_inv @ X.T
    e0 = M @ y
    e1 = M @ (W @ y)

    def neg_ll(rho):
        e = e0 - rho * e1
        sigma2 = float(e @ e) / n
        A = I - rho * W
        sign, logdetA = np.linalg.slogdet(A)
        if sign <= 0 or sigma2 <= 0:
            return 1e12
        return 0.5 * n * np.log(2 * np.pi * sigma2) - logdetA + 0.5 * n

    res = optimize.minimize_scalar(neg_ll, bounds=(-0.99, 0.99), method="bounded",
                                   options={"xatol": 1e-5})
    rho = float(res.x)
    Wy = W @ y
    # OLS of (y - rho Wy) on X
    y_star = y - rho * Wy
    beta = np.linalg.solve(X.T @ X, X.T @ y_star)
    e = y_star - X @ beta
    sigma2 = float(e @ e) / max(n - p - 1, 1)
    cov_beta = sigma2 * XtX_inv
    se_beta = np.sqrt(np.maximum(np.diag(cov_beta), 0.0))

    return RichResult(payload={
        "estimate": beta.tolist(),
        "se": se_beta.tolist(),
        "rho": rho,
        "sigma2": sigma2,
        "n": int(n),
        "method": "SAR lag (ML, concentrated log-likelihood)",
    })


def cheatsheet():
    return "sarla: SAR lag model (ML)"


# CANONICAL TEST
# Same X, y, W as sarre canonical. Expect rho ~ small (data has no lag
# structure) and beta ~ [intercept, slope].
