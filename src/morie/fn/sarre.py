"""Spatial autoregressive error model (SAR error, ML)."""
import numpy as np
from scipy import optimize
from ._richresult import RichResult

__all__ = ["spatial_ar_error"]


def spatial_ar_error(x, y, w):
    """
    SAR error model:
        Y = X beta + u,    u = lambda W u + eps,   eps ~ N(0, sigma2 I).

    Concentrated log-likelihood in lambda (Anselin 1988; Schabenberger
    & Gotway 2005, Ch 7):
        ll(lambda) = -n/2 log(2 pi sigma2_hat) + log|I - lambda W| - n/2
    with sigma2_hat = e' A' A e / n,  A = I - lambda W,  beta_hat from
    GLS on the transformed system  A y = A X beta + eps.

    Parameters
    ----------
    x : array-like, shape (n, p) -- design matrix (intercept explicit).
    y : array-like, shape (n,)
    w : array-like, shape (n, n) -- row-standardised spatial weights.

    Returns
    -------
    RichResult with payload: estimate (beta), se, lambda, sigma2,
        n, method.
    """
    X = np.asarray(x, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y, dtype=float).ravel()
    W = np.asarray(w, dtype=float)
    n, p = X.shape
    if y.size != n:
        raise ValueError(f"len(y)={y.size} != n={n}")
    if W.shape != (n, n):
        raise ValueError(f"w must be ({n},{n})")
    I = np.eye(n)

    def neg_ll(lam):
        A = I - lam * W
        AX = A @ X; Ay = A @ y
        try:
            beta = np.linalg.solve(AX.T @ AX, AX.T @ Ay)
        except np.linalg.LinAlgError:
            return 1e12
        e = Ay - AX @ beta
        sigma2 = float(e @ e) / n
        try:
            sign, logdetA = np.linalg.slogdet(A)
        except np.linalg.LinAlgError:
            return 1e12
        if sign <= 0 or sigma2 <= 0:
            return 1e12
        return 0.5 * n * np.log(2 * np.pi * sigma2) - logdetA + 0.5 * n

    # Search lambda inside (1/min_eig, 1/max_eig) -- approximate via (-0.99, 0.99)
    res = optimize.minimize_scalar(neg_ll, bounds=(-0.99, 0.99), method="bounded",
                                   options={"xatol": 1e-5})
    lam = float(res.x)
    A = I - lam * W
    AX = A @ X; Ay = A @ y
    beta = np.linalg.solve(AX.T @ AX, AX.T @ Ay)
    e = Ay - AX @ beta
    sigma2 = float(e @ e) / max(n - p, 1)
    cov_beta = sigma2 * np.linalg.inv(AX.T @ AX)
    se_beta = np.sqrt(np.maximum(np.diag(cov_beta), 0.0))

    return RichResult(payload={
        "estimate": beta.tolist(),
        "se": se_beta.tolist(),
        "lambda": lam,
        "sigma2": sigma2,
        "n": int(n),
        "method": "SAR error (ML, concentrated log-likelihood)",
    })


def cheatsheet():
    return "sarre: SAR error model (ML)"


# CANONICAL TEST
# X = column of ones + coord, y = 1 + 2*coord + small spatial error,
# W = path-graph row-standardised (5x5)
# Expect lambda in (-1, 1) and beta ~ [1, 2].
