# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Average derivative estimation."""

from __future__ import annotations

import numpy as np
from scipy import stats


def avgde(
    y: np.ndarray,
    X: np.ndarray,
    *,
    bandwidth: float | None = None,
    kernel: str = "gaussian",
) -> dict:
    r"""
    Average derivative estimation (ADE).

    Estimates :math:`\delta = E[\nabla m(X)]` where
    :math:`m(x) = E[Y \mid X = x]`. Uses the Hardle-Stoker (1989)
    density-weighted estimator:

    .. math::

        \hat{\delta}_j = -\frac{2}{n} \sum_{i=1}^{n}
        Y_i \, \hat{f}'_j(X_i) / \hat{f}(X_i)

    which avoids direct estimation of the regression derivative by
    exploiting integration by parts. For the univariate case
    (:math:`p = 1`), this simplifies to the Powell-Stock-Stoker (1989)
    estimator.

    Parameters
    ----------
    y : np.ndarray
        Response vector (n,).
    X : np.ndarray
        Covariate(s), (n,) or (n, p).
    bandwidth : float or None
        Kernel bandwidth. If None, Silverman's rule.
    kernel : str
        Kernel: ``'gaussian'``, ``'epanechnikov'``, ``'uniform'``.

    Returns
    -------
    dict
        Keys: ``avg_derivative`` (length-p array or scalar),
        ``se`` (standard errors), ``t_stat``, ``pval``,
        ``bandwidth``, ``n_obs``.

    References
    ----------
    Powell, J. L., Stock, J. H., & Stoker, T. M. (1989). Semiparametric
        estimation of index coefficients. Econometrica, 57(6), 1403-1430.
    Hardle, W. & Stoker, T. M. (1989). Investigating smooth multiple
        regression by the method of average derivatives. JASA, 84, 986-995.
    Horowitz, J. L. (2009). Semiparametric and Nonparametric Methods in
        Econometrics. Springer. Chapter 4.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    univariate = X.ndim == 1
    if univariate:
        X = X.reshape(-1, 1)
    n, p = X.shape
    if y.shape[0] != n:
        raise ValueError(f"y length {y.shape[0]} != X rows {n}.")
    if n < 10:
        raise ValueError("Need at least 10 observations.")

    from morie.fn.nwker import _get_kernel, _silverman_bw
    k_fn = _get_kernel(kernel)

    if bandwidth is None:
        bws = np.array([_silverman_bw(X[:, j]) for j in range(p)])
        h = float(np.mean(bws))
    else:
        h = bandwidth

    delta = np.zeros(p)

    for j in range(p):
        xj = X[:, j]
        diff = xj[:, None] - xj[None, :]
        u = diff / h
        k_vals = k_fn(u) / h

        f_hat = k_vals.mean(axis=1)

        k_prime = -u * k_fn(u) / h
        f_prime = k_prime.mean(axis=1) / h

        mask = f_hat > 1e-15
        ratio = np.zeros(n)
        ratio[mask] = f_prime[mask] / f_hat[mask]

        delta[j] = -2.0 * np.mean(y * ratio)

    se = np.full(p, np.nan)
    for j in range(p):
        xj = X[:, j]
        diff = xj[:, None] - xj[None, :]
        u = diff / h
        k_vals = k_fn(u) / h
        f_hat = k_vals.mean(axis=1)
        k_prime = -u * k_fn(u) / h
        f_prime = k_prime.mean(axis=1) / h
        mask = f_hat > 1e-15
        ratio = np.zeros(n)
        ratio[mask] = f_prime[mask] / f_hat[mask]
        psi_i = -2.0 * y * ratio - delta[j]
        se[j] = float(np.std(psi_i, ddof=1) / np.sqrt(n))

    t_stat = delta / np.where(se > 0, se, np.inf)
    pval = 2 * stats.norm.sf(np.abs(t_stat))

    if univariate:
        delta, se, t_stat, pval = delta[0], se[0], t_stat[0], pval[0]

    return {
        "avg_derivative": float(delta) if np.ndim(delta) == 0 else delta.tolist(),
        "se": float(se) if np.ndim(se) == 0 else se.tolist(),
        "t_stat": float(t_stat) if np.ndim(t_stat) == 0 else t_stat.tolist(),
        "pval": float(pval) if np.ndim(pval) == 0 else pval.tolist(),
        "bandwidth": h,
        "n_obs": n,
    }


avgde_fn = avgde


def cheatsheet() -> str:
    return "avgde({y, X}) -> Average derivative estimation."
