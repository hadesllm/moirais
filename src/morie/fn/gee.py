# morie.fn — function file (hadesllm/morie)
"""Generalized estimating equations (GEE)."""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ._containers import RegressionResult


def gee_regression(
    y: np.ndarray,
    X: np.ndarray,
    clusters: np.ndarray,
    *,
    family: str = "gaussian",
    corr_structure: str = "exchangeable",
    add_intercept: bool = True,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> RegressionResult:
    """Generalized estimating equations with sandwich (robust) SE.

    Parameters
    ----------
    y : (n,) response
    X : (n, p) predictors
    clusters : (n,) cluster/group identifiers
    family : str
        'gaussian' (identity link) or 'binomial' (logit link).
    corr_structure : str
        'independence' or 'exchangeable'.
    add_intercept : bool
    max_iter : int
    tol : float

    Returns
    -------
    RegressionResult
        With sandwich (robust) standard errors.

    References
    ----------
    Liang, K.-Y. & Zeger, S. L. (1986). Longitudinal data analysis using
    generalized linear models. *Biometrika*, 73(1), 13--22.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    clusters = np.asarray(clusters).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p_raw = X.shape
    if add_intercept:
        X = np.column_stack([np.ones(n), X])
    k = X.shape[1]

    from scipy import special

    def _link(eta):
        if family == "binomial":
            return special.expit(eta)
        return eta

    def _var(mu):
        if family == "binomial":
            return mu * (1 - mu) + 1e-12
        return np.ones_like(mu)

    unique_cl = np.unique(clusters)

    beta = np.zeros(k)
    for _ in range(max_iter):
        eta = X @ beta
        mu = _link(eta)
        v = _var(mu)

        if family == "binomial":
            d_mu = mu * (1 - mu) + 1e-12
        else:
            d_mu = np.ones(n)

        score = np.zeros(k)
        meat = np.zeros((k, k))
        bread = np.zeros((k, k))

        for cl in unique_cl:
            idx = np.where(clusters == cl)[0]
            ni = len(idx)
            Xi = X[idx]
            ri = y[idx] - mu[idx]
            Di = Xi * d_mu[idx, None]
            Vi_diag = v[idx]

            if corr_structure == "exchangeable" and ni > 1:
                alpha_est = 0.0
                for ii in range(ni):
                    for jj in range(ii + 1, ni):
                        alpha_est += ri[ii] * ri[jj] / np.sqrt(Vi_diag[ii] * Vi_diag[jj])
                n_pairs = ni * (ni - 1) / 2
                alpha_est /= (n_pairs + 1e-12)
                alpha_est = np.clip(alpha_est, -0.99, 0.99)
                R = np.eye(ni) * (1 - alpha_est) + alpha_est
                Vi = np.diag(np.sqrt(Vi_diag)) @ R @ np.diag(np.sqrt(Vi_diag))
            else:
                Vi = np.diag(Vi_diag)

            try:
                Vi_inv = np.linalg.inv(Vi + np.eye(ni) * 1e-10)
            except np.linalg.LinAlgError:
                Vi_inv = np.diag(1.0 / (Vi_diag + 1e-12))

            score += Di.T @ Vi_inv @ ri
            bread += Di.T @ Vi_inv @ Di
            Si = Di.T @ Vi_inv @ ri
            meat += np.outer(Si, Si)

        try:
            bread_inv = np.linalg.inv(bread + np.eye(k) * 1e-10)
        except np.linalg.LinAlgError:
            break
        beta_new = beta + bread_inv @ score
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    sandwich = bread_inv @ meat @ bread_inv
    se_arr = np.sqrt(np.diag(sandwich).clip(0))
    z_vals = beta / (se_arr + 1e-300)
    p_vals = 2.0 * _st.norm.sf(np.abs(z_vals))

    mu_f = _link(X @ beta)

    names = (["(Intercept)"] if add_intercept else []) + [
        f"x{j}" for j in range(p_raw)
    ]
    return RegressionResult(
        method=f"GEE ({family}, {corr_structure})",
        coefficients={nm: float(b) for nm, b in zip(names, beta)},
        se={nm: float(s) for nm, s in zip(names, se_arr)},
        p_values={nm: float(pv) for nm, pv in zip(names, p_vals)},
        fitted=mu_f,
        residuals=y - mu_f,
        n=n,
        k=k - (1 if add_intercept else 0),
        extra={"n_clusters": len(unique_cl), "corr_structure": corr_structure},
    )


gee = gee_regression


def cheatsheet() -> str:
    return "gee_regression({}) -> GEE with sandwich standard errors."
