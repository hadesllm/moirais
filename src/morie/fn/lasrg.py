# morie.fn — function file (hadesllm/morie)
"""LASSO regression via coordinate descent."""

from __future__ import annotations

import numpy as np

from ._containers import RegressionResult


def _soft_threshold(z: float, lam: float) -> float:
    if z > lam:
        return z - lam
    elif z < -lam:
        return z + lam
    return 0.0


def lasso_regression(
    y: np.ndarray,
    X: np.ndarray,
    *,
    lam: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> RegressionResult:
    r"""LASSO via coordinate descent on standardised predictors.

    Minimises :math:`\\frac{1}{2n}\\|y - X\\beta\\|^2 + \\lambda \\|\\beta\\|_1`.

    Parameters
    ----------
    y : (n,) array
    X : (n, p) array
    lam : float
        L1 penalty strength.
    max_iter : int
    tol : float

    Returns
    -------
    RegressionResult

    References
    ----------
    Tibshirani, R. (1996). Regression shrinkage and selection via the lasso.
    *JRSS-B*, 58(1), 267--288.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape

    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0, ddof=0)
    x_std[x_std == 0] = 1.0
    Xs = (X - x_mean) / x_std
    y_mean = y.mean()
    yc = y - y_mean

    beta = np.zeros(p)
    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            r_j = yc - Xs @ beta + Xs[:, j] * beta[j]
            z_j = Xs[:, j] @ r_j / n
            beta[j] = _soft_threshold(z_j, lam)
        if np.max(np.abs(beta - beta_old)) < tol:
            break

    beta_orig = beta / x_std
    intercept = y_mean - x_mean @ beta_orig
    fitted = X @ beta_orig + intercept
    resid = y - fitted
    ss_res = float(resid @ resid)
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    n_nonzero = int(np.sum(beta != 0))

    names = ["(Intercept)"] + [f"x{j}" for j in range(p)]
    coefs = [intercept] + beta_orig.tolist()

    return RegressionResult(
        method=f"LASSO (lam={lam})",
        coefficients={nm: float(b) for nm, b in zip(names, coefs)},
        se={nm: float("nan") for nm in names},
        p_values={nm: float("nan") for nm in names},
        r_squared=r2,
        residuals=resid,
        fitted=fitted,
        n=n,
        k=p,
        extra={"lam": lam, "n_nonzero": n_nonzero},
    )


lasrg = lasso_regression


def cheatsheet() -> str:
    return "lasso_regression({}) -> LASSO via coordinate descent."
