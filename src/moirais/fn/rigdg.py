# moirais.fn — function file (hadesllm/moirais)
"""Ridge regression (closed-form)."""

from __future__ import annotations

import numpy as np

from ._containers import RegressionResult


def ridge_regression(
    y: np.ndarray,
    X: np.ndarray,
    *,
    lam: float = 1.0,
    add_intercept: bool = True,
) -> RegressionResult:
    """Ridge regression: :math:`\\hat{\\beta} = (X^\\top X + \\lambda I)^{-1} X^\\top y`.

    Standardises predictors internally; returns coefficients on the
    original scale.

    Parameters
    ----------
    y : (n,) array
    X : (n, p) array
    lam : float
        Regularization parameter (>= 0).
    add_intercept : bool

    Returns
    -------
    RegressionResult

    References
    ----------
    Hoerl, A. E. & Kennard, R. W. (1970). Ridge regression: biased estimation
    for nonorthogonal problems. *Technometrics*, 12(1), 55--67.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    if lam < 0:
        raise ValueError("lam must be non-negative.")

    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0, ddof=0)
    x_std[x_std == 0] = 1.0
    Xs = (X - x_mean) / x_std
    y_mean = y.mean()
    yc = y - y_mean

    XtX = Xs.T @ Xs
    beta_s = np.linalg.solve(XtX + lam * np.eye(p), Xs.T @ yc)

    beta_orig = beta_s / x_std
    intercept = y_mean - x_mean @ beta_orig

    fitted = X @ beta_orig + intercept
    resid = y - fitted
    ss_res = float(resid @ resid)
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    eff_df = np.trace(Xs @ np.linalg.inv(XtX + lam * np.eye(p)) @ Xs.T)
    gcv = (ss_res / n) / (1.0 - eff_df / n) ** 2 if n > eff_df else float("inf")

    names = ["(Intercept)"] + [f"x{j}" for j in range(p)]
    coefs = [intercept] + beta_orig.tolist()

    return RegressionResult(
        method=f"Ridge (lam={lam})",
        coefficients={nm: float(b) for nm, b in zip(names, coefs)},
        se={nm: float("nan") for nm in names},
        p_values={nm: float("nan") for nm in names},
        r_squared=r2,
        residuals=resid,
        fitted=fitted,
        n=n,
        k=p,
        extra={"lam": lam, "eff_df": float(eff_df), "gcv": float(gcv)},
    )


rigdg = ridge_regression


def cheatsheet() -> str:
    return "ridge_regression({}) -> Ridge regression (closed-form with lambda)."
