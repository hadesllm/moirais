# morie.fn -- function file (hadesllm/morie)
"""L1/Elastic net regularization path. 'If you're guilty, you're dead.' -- Punisher"""

from __future__ import annotations

import numpy as np

from ._containers import RegressionResult


def penalty_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 1.0,
    l1_ratio: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> RegressionResult:
    """Fit elastic net regression via coordinate descent.

    Minimises :math:`\\frac{1}{2n}||y - X\\beta||_2^2 + \\alpha[l_1 ||\\beta||_1 + \\frac{1-l_1}{2}||\\beta||_2^2]`.

    When ``l1_ratio=1``, this is Lasso; when ``l1_ratio=0``, Ridge.

    Parameters
    ----------
    X : np.ndarray
        (n x p) predictor matrix.
    y : np.ndarray
        (n,) response vector.
    alpha : float
        Overall regularization strength.
    l1_ratio : float
        Mixing parameter in [0, 1].
    max_iter : int
        Maximum coordinate descent iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    RegressionResult
        Coefficients, standard errors (from KKT), R-squared.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    n, p = X.shape
    if len(y) != n:
        raise ValueError("X and y must have same n")

    X_mean = X.mean(axis=0)
    y_mean = float(y.mean())
    Xc = X - X_mean
    yc = y - y_mean

    col_norms = (Xc**2).sum(axis=0)
    beta = np.zeros(p)

    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            r = yc - Xc @ beta + Xc[:, j] * beta[j]
            rho = Xc[:, j] @ r / n
            l1 = alpha * l1_ratio
            l2 = alpha * (1 - l1_ratio)
            if abs(rho) <= l1:
                beta[j] = 0.0
            else:
                beta[j] = (np.sign(rho) * (abs(rho) - l1)) / (col_norms[j] / n + l2)
        if np.max(np.abs(beta - beta_old)) < tol:
            break

    intercept = y_mean - X_mean @ beta
    fitted = X @ beta + intercept
    residuals = y - fitted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    coefs = {f"x{j}": float(beta[j]) for j in range(p)}
    coefs["intercept"] = float(intercept)

    se_approx = {}
    mse = ss_res / max(n - np.sum(beta != 0) - 1, 1)
    for j in range(p):
        if col_norms[j] > 0:
            se_approx[f"x{j}"] = float(np.sqrt(mse / col_norms[j]))
        else:
            se_approx[f"x{j}"] = 0.0
    se_approx["intercept"] = float(np.sqrt(mse / n))

    from scipy import stats

    p_vals = {}
    for key in coefs:
        if se_approx.get(key, 0) > 0:
            t_stat = coefs[key] / se_approx[key]
            p_vals[key] = float(2 * stats.t.sf(abs(t_stat), max(n - p - 1, 1)))
        else:
            p_vals[key] = 1.0

    return RegressionResult(
        method=f"elastic_net (alpha={alpha}, l1_ratio={l1_ratio})",
        coefficients=coefs,
        se=se_approx,
        p_values=p_vals,
        r_squared=r2,
        residuals=residuals,
        fitted=fitted,
        n=n,
        k=p + 1,
        extra={"alpha": alpha, "l1_ratio": l1_ratio, "n_nonzero": int(np.sum(beta != 0))},
    )


punsh = penalty_regression


def cheatsheet() -> str:
    return "penalty_regression({}) -> L1/Elastic net regularization path. 'If you're guilty, you'r"
