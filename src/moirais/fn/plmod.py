# moirais.fn — function file (hadesllm/moirais)
"""Partially linear model (Robinson 1988)."""

from __future__ import annotations

import numpy as np
from scipy import stats


def plmod(
    y: np.ndarray,
    d: np.ndarray,
    x: np.ndarray,
    *,
    bandwidth: float | None = None,
    kernel: str = "gaussian",
) -> dict:
    r"""
    Partially linear model via Robinson's (1988) double-residual method.

    The model is:

    .. math::

        Y_i = D_i^\top \theta + g(X_i) + \varepsilon_i

    where :math:`\theta` is a finite-dimensional parameter and :math:`g(\cdot)`
    is an unknown smooth function. Robinson's procedure:

    1. Estimate :math:`E[Y \mid X]` and :math:`E[D \mid X]` nonparametrically
       (Nadaraya-Watson).
    2. Regress :math:`\tilde{Y}_i = Y_i - \hat{E}[Y \mid X_i]` on
       :math:`\tilde{D}_i = D_i - \hat{E}[D \mid X_i]` via OLS.

    Parameters
    ----------
    y : np.ndarray
        Outcome vector (n,).
    d : np.ndarray
        Treatment/parametric variable(s). Shape (n,) or (n, p).
    x : np.ndarray
        Nonparametric covariate (n,). Currently 1-d.
    bandwidth : float or None
        Bandwidth for NW regression. If None, Silverman's rule.
    kernel : str
        Kernel: ``'gaussian'``, ``'epanechnikov'``, ``'uniform'``.

    Returns
    -------
    dict
        Keys: ``theta`` (parameter estimate), ``se`` (standard error),
        ``t_stat``, ``pval``, ``ci_lower``, ``ci_upper``, ``bandwidth``,
        ``n_obs``.

    Raises
    ------
    ValueError
        If array dimensions are incompatible.

    References
    ----------
    Robinson, P. M. (1988). Root-N-consistent semiparametric regression.
        Econometrica, 56(4), 931-954.
    Horowitz, J. L. (2009). Semiparametric and Nonparametric Methods in
        Econometrics. Springer. Chapter 3.
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    d = np.asarray(d, dtype=float)
    if d.ndim == 1:
        d = d.reshape(-1, 1)
    n = y.shape[0]
    if x.shape[0] != n or d.shape[0] != n:
        raise ValueError("y, d, and x must have the same number of observations.")
    if n < 10:
        raise ValueError("Need at least 10 observations.")

    from moirais.fn.nwker import nwker

    bw_kw = {"bandwidth": bandwidth, "kernel": kernel}
    e_y_x = nwker(x, y, **bw_kw)["y_hat"]
    used_bw = nwker(x, y, **bw_kw)["bandwidth"]

    p = d.shape[1]
    e_d_x = np.empty_like(d)
    for j in range(p):
        e_d_x[:, j] = nwker(x, d[:, j], **bw_kw)["y_hat"]

    y_tilde = y - e_y_x
    d_tilde = d - e_d_x

    dtd = d_tilde.T @ d_tilde
    if np.linalg.matrix_rank(dtd) < p:
        raise ValueError("Residualized treatment matrix is singular.")

    theta = np.linalg.solve(dtd, d_tilde.T @ y_tilde)
    resid = y_tilde - d_tilde @ theta
    sigma2 = float(np.sum(resid ** 2) / (n - p))
    var_theta = sigma2 * np.linalg.inv(dtd)

    se = np.sqrt(np.diag(var_theta))
    t_stat = theta / se
    pval = 2 * stats.t.sf(np.abs(t_stat), df=n - p)
    ci_lower = theta - 1.96 * se
    ci_upper = theta + 1.96 * se

    if p == 1:
        theta, se, t_stat, pval = theta[0], se[0], t_stat[0], pval[0]
        ci_lower, ci_upper = ci_lower[0], ci_upper[0]

    return {
        "theta": float(theta) if np.ndim(theta) == 0 else theta.tolist(),
        "se": float(se) if np.ndim(se) == 0 else se.tolist(),
        "t_stat": float(t_stat) if np.ndim(t_stat) == 0 else t_stat.tolist(),
        "pval": float(pval) if np.ndim(pval) == 0 else pval.tolist(),
        "ci_lower": float(ci_lower) if np.ndim(ci_lower) == 0 else ci_lower.tolist(),
        "ci_upper": float(ci_upper) if np.ndim(ci_upper) == 0 else ci_upper.tolist(),
        "bandwidth": used_bw,
        "n_obs": n,
    }


plmod_fn = plmod


def cheatsheet() -> str:
    return "plmod({y, d, x}) -> Partially linear model (Robinson 1988)."
