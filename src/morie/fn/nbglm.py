# morie.fn -- function file (hadesllm/morie)
"""Negative binomial GLM via IRLS."""

from __future__ import annotations

import numpy as np
from scipy import special
from scipy import stats as _st

from ._containers import RegressionResult


def negbin_glm(
    y: np.ndarray,
    X: np.ndarray,
    *,
    alpha: float = 1.0,
    add_intercept: bool = True,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> RegressionResult:
    r"""Negative binomial GLM (NB2) via IRLS with fixed dispersion.

    Log-link: :math:`\\log(\\mu) = X \\beta`.  Variance function
    :math:`V(\\mu) = \\mu + \\alpha \\mu^2` (NB2 parameterisation).

    Parameters
    ----------
    y : (n,) non-negative integer counts
    X : (n, p) predictors
    alpha : float
        Overdispersion parameter (> 0). alpha=0 reduces to Poisson.
    add_intercept : bool
    max_iter : int
    tol : float

    Returns
    -------
    RegressionResult

    References
    ----------
    Cameron, A. C. & Trivedi, P. K. (2013). *Regression Analysis of Count
    Data* (2nd ed.). Cambridge.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p_raw = X.shape
    if add_intercept:
        X = np.column_stack([np.ones(n), X])
    k = X.shape[1]

    beta = np.zeros(k)
    beta[0] = np.log(np.mean(y) + 1e-8)

    for _ in range(max_iter):
        eta = X @ beta
        mu = np.exp(np.clip(eta, -20, 20))
        v = mu + alpha * mu ** 2
        W = mu ** 2 / (v + 1e-12)
        z = eta + (y - mu) / (mu + 1e-12)
        XtWX = (X * W[:, None]).T @ X
        XtWz = (X * W[:, None]).T @ z
        try:
            beta_new = np.linalg.solve(XtWX + np.eye(k) * 1e-10, XtWz)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    mu_f = np.exp(np.clip(X @ beta, -20, 20))
    r_nb = 1.0 / (alpha + 1e-12)
    ll = float(np.sum(
        special.gammaln(y + r_nb)
        - special.gammaln(r_nb)
        - special.gammaln(y + 1)
        + r_nb * np.log(r_nb / (r_nb + mu_f + 1e-12))
        + y * np.log(mu_f / (r_nb + mu_f + 1e-12) + 1e-300)
    ))
    aic = -2.0 * ll + 2 * k

    v_f = mu_f + alpha * mu_f ** 2
    W_f = mu_f ** 2 / (v_f + 1e-12)
    XtWX = (X * W_f[:, None]).T @ X
    try:
        cov = np.linalg.inv(XtWX)
        se_arr = np.sqrt(np.diag(cov).clip(0))
    except np.linalg.LinAlgError:
        se_arr = np.full(k, float("nan"))

    z_vals = beta / (se_arr + 1e-300)
    p_vals = 2.0 * _st.norm.sf(np.abs(z_vals))

    names = (["(Intercept)"] if add_intercept else []) + [
        f"x{j}" for j in range(p_raw)
    ]
    return RegressionResult(
        method=f"NegBin GLM (alpha={alpha})",
        coefficients={nm: float(b) for nm, b in zip(names, beta)},
        se={nm: float(s) for nm, s in zip(names, se_arr)},
        p_values={nm: float(pv) for nm, pv in zip(names, p_vals)},
        fitted=mu_f,
        residuals=y - mu_f,
        n=n,
        k=k - (1 if add_intercept else 0),
        extra={"alpha": alpha, "aic": aic, "log_likelihood": ll},
    )


nbglm = negbin_glm


def cheatsheet() -> str:
    return "negbin_glm({}) -> Negative binomial GLM via IRLS."
