"""Zero-inflated Poisson regression."""

from __future__ import annotations

import numpy as np
from scipy import special

from ._containers import RegressionResult


def zero_inflated_poisson(
    y: np.ndarray,
    X: np.ndarray,
    *,
    Z: np.ndarray | None = None,
    add_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> RegressionResult:
    """Zero-inflated Poisson (ZIP) via EM algorithm.

    Mixture of a point mass at zero (logistic inflation) and a Poisson
    count model.  If *Z* is not given, the inflation equation uses the
    same predictors as the count equation.

    Parameters
    ----------
    y : (n,) non-negative integer counts
    X : (n, p) count-model predictors
    Z : (n, q) inflation predictors (default: same as X)
    add_intercept : bool
    max_iter : int
    tol : float

    Returns
    -------
    RegressionResult

    References
    ----------
    Lambert, D. (1992). Zero-inflated Poisson regression, with an
    application to defects in manufacturing. *Technometrics*, 34(1), 1--14.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p_raw = X.shape
    if Z is None:
        Z = X.copy()
    else:
        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)

    if add_intercept:
        X = np.column_stack([np.ones(n), X])
        Z = np.column_stack([np.ones(n), Z])
    kx = X.shape[1]
    kz = Z.shape[1]

    beta = np.zeros(kx)
    beta[0] = np.log(np.mean(y) + 1e-8)
    gamma = np.zeros(kz)

    for _ in range(max_iter):
        eta = X @ beta
        mu = np.exp(np.clip(eta, -20, 20))
        psi = special.expit(Z @ gamma)

        p0_poisson = np.exp(-mu)
        denom = psi + (1 - psi) * p0_poisson
        denom = np.maximum(denom, 1e-300)
        w = np.where(y == 0, psi / denom, 0.0)

        wt_count = 1.0 - w
        XtWX = (X * (wt_count * mu)[:, None]).T @ X
        z_wls = eta + (y - mu) / (mu + 1e-12)
        XtWz = (X * (wt_count * mu)[:, None]).T @ z_wls
        try:
            beta_new = np.linalg.solve(XtWX + np.eye(kx) * 1e-10, XtWz)
        except np.linalg.LinAlgError:
            break

        for _irls in range(20):
            psi_irls = special.expit(Z @ gamma)
            psi_irls = np.clip(psi_irls, 1e-8, 1 - 1e-8)
            W_z = psi_irls * (1 - psi_irls) + 1e-12
            z_z = Z @ gamma + (w - psi_irls) / W_z
            ZtWZ = (Z * W_z[:, None]).T @ Z
            ZtWz = (Z * W_z[:, None]).T @ z_z
            try:
                gamma_new = np.linalg.solve(ZtWZ + np.eye(kz) * 1e-10, ZtWz)
            except np.linalg.LinAlgError:
                break
            if np.max(np.abs(gamma_new - gamma)) < tol:
                gamma = gamma_new
                break
            gamma = gamma_new

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    mu_f = np.exp(np.clip(X @ beta, -20, 20))
    psi_f = special.expit(Z @ gamma)
    fitted = (1 - psi_f) * mu_f

    ll = 0.0
    for i in range(n):
        if y[i] == 0:
            ll += np.log(psi_f[i] + (1 - psi_f[i]) * np.exp(-mu_f[i]) + 1e-300)
        else:
            ll += np.log(1 - psi_f[i] + 1e-300) + y[i] * np.log(mu_f[i] + 1e-300) - mu_f[i] - float(special.gammaln(y[i] + 1))
    aic = -2 * ll + 2 * (kx + kz)

    count_names = (["count_(Intercept)"] if add_intercept else []) + [
        f"count_x{j}" for j in range(p_raw)
    ]
    inflate_names = (["inflate_(Intercept)"] if add_intercept else []) + [
        f"inflate_z{j}" for j in range(Z.shape[1] - (1 if add_intercept else 0))
    ]
    all_names = count_names + inflate_names
    all_coefs = np.concatenate([beta, gamma])

    return RegressionResult(
        method="Zero-Inflated Poisson",
        coefficients={nm: float(b) for nm, b in zip(all_names, all_coefs)},
        se={nm: float("nan") for nm in all_names},
        p_values={nm: float("nan") for nm in all_names},
        fitted=fitted,
        residuals=y - fitted,
        n=n,
        k=kx + kz,
        extra={
            "log_likelihood": float(ll),
            "aic": aic,
            "n_zeros": int(np.sum(y == 0)),
            "psi_mean": float(np.mean(psi_f)),
        },
    )


zinfl = zero_inflated_poisson


def cheatsheet() -> str:
    return "zero_inflated_poisson({}) -> ZIP regression via EM."
