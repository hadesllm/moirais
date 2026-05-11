"""Transformation model MLE."""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import minimize


def trmle(
    y: np.ndarray,
    X: np.ndarray,
    *,
    n_basis: int = 5,
) -> dict:
    r"""
    Semiparametric transformation model via profile MLE.

    Fits the model :math:`\Lambda(Y) = X'\beta + \varepsilon` where
    the transformation :math:`\Lambda` is approximated by a monotone
    Bernstein polynomial basis.

    Maximises the profile log-likelihood over :math:`\beta` and the
    basis coefficients jointly.

    Parameters
    ----------
    y : np.ndarray
        Response (n,).
    X : np.ndarray
        Covariates (n, p).
    n_basis : int
        Number of Bernstein polynomial basis functions.

    Returns
    -------
    dict
        ``beta``, ``se``, ``t_stat``, ``pval``, ``basis_coefs``,
        ``log_likelihood``, ``n_obs``.

    References
    ----------
    Horowitz, J. L. (1996). Semiparametric estimation of a regression
        model with an unknown transformation of the dependent variable.
        Econometrica, 64, 103-137.
    Horowitz (2009). Ch 6.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    if y.shape[0] != n:
        raise ValueError("y and X must have same n.")
    if n < p + n_basis + 2:
        raise ValueError("Insufficient observations.")

    y_lo, y_hi = y.min(), y.max()
    y_range = y_hi - y_lo if y_hi > y_lo else 1.0
    y_scaled = (y - y_lo) / y_range

    from scipy.special import comb as _comb

    def bernstein_basis(u, k, deg):
        return _comb(deg, k) * u**k * (1 - u) ** (deg - k)

    deg = n_basis - 1
    B = np.column_stack([bernstein_basis(y_scaled, k, deg) for k in range(n_basis)])

    def neg_ll(params):
        beta = params[:p]
        gamma_raw = params[p:]
        gamma = np.cumsum(np.exp(gamma_raw))

        lam_y = B @ gamma
        resid = lam_y - X @ beta
        sigma2 = np.var(resid) + 1e-10
        ll = -0.5 * n * np.log(sigma2) - 0.5 * np.sum(resid**2) / sigma2

        dlam = np.zeros(n)
        for k in range(n_basis):
            if deg > 0:
                if k < deg:
                    dlam += gamma[k] * deg * (
                        bernstein_basis(y_scaled, max(k - 1, 0), max(deg - 1, 0)) * (1 if k > 0 else 0)
                        - bernstein_basis(y_scaled, k, max(deg - 1, 0))
                    ) if deg > 0 else 0
            dlam_k = gamma[k] * deg * _comb(deg - 1, min(k, deg - 1)) * (
                y_scaled ** max(k - 1, 0) * (1 - y_scaled) ** max(deg - 1 - max(k - 1, 0), 0)
            ) / y_range if deg > 0 else gamma[k] / y_range
            dlam += dlam_k

        dlam = np.maximum(dlam, 1e-15)
        ll += np.sum(np.log(dlam))
        return -ll

    b0_beta = np.linalg.lstsq(X, y_scaled, rcond=None)[0]
    b0_gamma = np.zeros(n_basis)
    params0 = np.concatenate([b0_beta, b0_gamma])

    res = minimize(neg_ll, params0, method="L-BFGS-B",
                   options={"maxiter": 300, "ftol": 1e-8})

    beta = res.x[:p]
    gamma = np.cumsum(np.exp(res.x[p:]))

    lam_y = B @ gamma
    resid = lam_y - X @ beta
    sigma2 = float(np.var(resid) + 1e-10)

    try:
        from scipy.optimize import approx_fprime
        H = np.zeros((p, p))
        for i in range(p):
            ei = np.zeros(len(res.x))
            ei[i] = 1e-5
            g1 = approx_fprime(res.x, neg_ll, 1e-5)
            H[i, :] = approx_fprime(res.x, neg_ll, 1e-5)[:p]
        cov = sigma2 * np.linalg.pinv(X.T @ X / n) / n
    except Exception:
        cov = sigma2 * np.linalg.pinv(X.T @ X / n) / n

    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    t_stat = beta / np.where(se > 0, se, np.inf)
    pval = 2 * stats.norm.sf(np.abs(t_stat))

    return {
        "beta": beta.tolist(),
        "se": se.tolist(),
        "t_stat": t_stat.tolist(),
        "pval": pval.tolist(),
        "basis_coefs": gamma.tolist(),
        "log_likelihood": -float(res.fun),
        "n_obs": n,
    }


trmle_fn = trmle


def cheatsheet() -> str:
    return "trmle({y, X}) -> Transformation model semiparametric MLE."
