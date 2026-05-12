# morie.fn -- function file (hadesllm/morie)
"""Hurdle model (two-part: logistic + truncated Poisson)."""

from __future__ import annotations

import numpy as np
from scipy import special

from ._containers import RegressionResult


def hurdle_model(
    y: np.ndarray,
    X: np.ndarray,
    *,
    add_intercept: bool = True,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> RegressionResult:
    """Two-part hurdle model: logistic zero/nonzero + truncated Poisson.

    Part 1 (hurdle): logistic regression for P(y > 0).
    Part 2 (count): zero-truncated Poisson for y | y > 0.

    Parameters
    ----------
    y : (n,) non-negative integer counts
    X : (n, p) predictors (used for both parts)
    add_intercept : bool
    max_iter : int
    tol : float

    Returns
    -------
    RegressionResult

    References
    ----------
    Mullahy, J. (1986). Specification and testing of some modified count
    data models. *J. Econometrics*, 33(3), 341--365.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p_raw = X.shape
    if add_intercept:
        X = np.column_stack([np.ones(n), X])
    k = X.shape[1]

    pos = y > 0
    d = pos.astype(float)
    y_pos = y[pos]
    X_pos = X[pos]

    gamma = np.zeros(k)
    for _ in range(max_iter):
        eta = X @ gamma
        mu = special.expit(eta)
        mu = np.clip(mu, 1e-8, 1 - 1e-8)
        W = mu * (1 - mu) + 1e-12
        z = eta + (d - mu) / W
        XtWX = (X * W[:, None]).T @ X
        XtWz = (X * W[:, None]).T @ z
        try:
            gamma_new = np.linalg.solve(XtWX + np.eye(k) * 1e-10, XtWz)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(gamma_new - gamma)) < tol:
            gamma = gamma_new
            break
        gamma = gamma_new

    beta = np.zeros(k)
    beta[0] = np.log(np.mean(y_pos))
    n_pos = len(y_pos)
    for _ in range(max_iter):
        eta_p = X_pos @ beta
        mu_p = np.exp(np.clip(eta_p, -20, 20))
        trunc_factor = 1.0 / (1.0 - np.exp(-mu_p) + 1e-300)
        y_adj = y_pos - mu_p * trunc_factor
        W_p = mu_p * trunc_factor + 1e-12
        z_p = eta_p + y_adj / W_p
        XtWX_p = (X_pos * W_p[:, None]).T @ X_pos
        XtWz_p = (X_pos * W_p[:, None]).T @ z_p
        try:
            beta_new = np.linalg.solve(XtWX_p + np.eye(k) * 1e-10, XtWz_p)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    pi_hat = special.expit(X @ gamma)
    mu_hat = np.exp(np.clip(X @ beta, -20, 20))
    fitted = pi_hat * mu_hat / (1 - np.exp(-mu_hat) + 1e-300)

    ll_zero = float(np.sum(d * np.log(pi_hat + 1e-300) + (1 - d) * np.log(1 - pi_hat + 1e-300)))
    ll_count = float(np.sum(
        y_pos * np.log(mu_p + 1e-300)
        - mu_p
        - np.array([float(special.gammaln(yi + 1)) for yi in y_pos])
        - np.log(1 - np.exp(-mu_p) + 1e-300)
    ))
    ll = ll_zero + ll_count

    hurdle_names = (["hurdle_(Intercept)"] if add_intercept else []) + [
        f"hurdle_x{j}" for j in range(p_raw)
    ]
    count_names = (["count_(Intercept)"] if add_intercept else []) + [
        f"count_x{j}" for j in range(p_raw)
    ]
    all_names = hurdle_names + count_names
    all_coefs = np.concatenate([gamma, beta])

    return RegressionResult(
        method="Hurdle (Logistic + Truncated Poisson)",
        coefficients={nm: float(b) for nm, b in zip(all_names, all_coefs)},
        se={nm: float("nan") for nm in all_names},
        p_values={nm: float("nan") for nm in all_names},
        fitted=fitted,
        residuals=y - fitted,
        n=n,
        k=2 * k,
        extra={
            "log_likelihood": float(ll),
            "n_zeros": int(np.sum(~pos)),
            "n_positive": n_pos,
        },
    )


hurds = hurdle_model


def cheatsheet() -> str:
    return "hurdle_model({}) -> Two-part hurdle model."
