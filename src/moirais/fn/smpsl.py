"""Sample selection correction (semiparametric Heckman)."""

from __future__ import annotations

import numpy as np
from scipy import stats


def smpsl(
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    d: np.ndarray,
    *,
    bandwidth: float | None = None,
    kernel: str = "gaussian",
) -> dict:
    r"""
    Semiparametric sample selection correction (Heckman-type).

    Two-stage procedure:

    1. Estimate the selection equation :math:`P(D=1|Z)` nonparametrically.
    2. Use the estimated propensity as a control function in the
       outcome equation: :math:`Y = X'\beta + \lambda(\hat{p}(Z)) + \varepsilon`.

    The correction function :math:`\lambda(\cdot)` is estimated via
    series approximation (polynomial in :math:`\hat{p}`).

    Parameters
    ----------
    y : np.ndarray
        Outcome (n,), observed only when d=1.
    X : np.ndarray
        Outcome covariates (n, p).
    Z : np.ndarray
        Selection covariates (n, q).
    d : np.ndarray
        Selection indicator (n,), binary {0, 1}.
    bandwidth : float or None
        Bandwidth for propensity estimation.
    kernel : str
        Kernel function.

    Returns
    -------
    dict
        ``beta``, ``se``, ``t_stat``, ``pval``, ``lambda_coefs``
        (correction polynomial), ``n_selected``, ``n_obs``.

    References
    ----------
    Heckman, J. J. (1979). Sample selection bias as a specification error.
        Econometrica, 47, 153-161.
    Horowitz (2009). Ch 7.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)
    d = np.asarray(d, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    n = d.shape[0]
    p = X.shape[1]
    if y.shape[0] != n or X.shape[0] != n or Z.shape[0] != n:
        raise ValueError("y, X, Z, d must have same n.")
    if not np.all(np.isin(d, [0, 1])):
        raise ValueError("d must be binary (0/1).")

    from moirais.fn.nwker import _get_kernel, _silverman_bw

    k_fn = _get_kernel(kernel)
    z_index = Z @ np.ones(Z.shape[1])
    h = bandwidth if bandwidth is not None else _silverman_bw(z_index)

    diff = z_index[:, None] - z_index[None, :]
    K = k_fn(diff / h)
    denom = K.sum(axis=1)
    denom = np.where(denom < 1e-15, 1.0, denom)
    p_hat = (K @ d) / denom
    p_hat = np.clip(p_hat, 0.01, 0.99)

    sel = d == 1
    n_sel = int(sel.sum())
    if n_sel < p + 3:
        raise ValueError(f"Only {n_sel} selected; need at least {p + 3}.")

    n_poly = min(3, n_sel // 5)
    P_basis = np.column_stack([p_hat[sel] ** k for k in range(1, n_poly + 1)])
    W = np.column_stack([X[sel], P_basis])

    WtW = W.T @ W
    try:
        theta = np.linalg.solve(WtW, W.T @ y[sel])
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(WtW, W.T @ y[sel], rcond=None)[0]

    beta = theta[:p]
    lam_coefs = theta[p:]

    resid = y[sel] - W @ theta
    df = max(n_sel - p - n_poly, 1)
    sigma2 = float(np.sum(resid**2) / df)
    try:
        cov = sigma2 * np.linalg.inv(WtW)
    except np.linalg.LinAlgError:
        cov = sigma2 * np.linalg.pinv(WtW)

    se = np.sqrt(np.maximum(np.diag(cov)[:p], 0.0))
    t_stat = beta / np.where(se > 0, se, np.inf)
    pval = 2 * stats.t.sf(np.abs(t_stat), df=df)

    return {
        "beta": beta.tolist(),
        "se": se.tolist(),
        "t_stat": t_stat.tolist(),
        "pval": pval.tolist(),
        "lambda_coefs": lam_coefs.tolist(),
        "n_selected": n_sel,
        "n_obs": n,
    }


smpsl_fn = smpsl


def cheatsheet() -> str:
    return "smpsl({y, X, Z, d}) -> Semiparametric sample selection correction."
