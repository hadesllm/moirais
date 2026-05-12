# morie.fn -- function file (hadesllm/morie)
"""DCC multivariate GARCH (Engle 2002)."""
from __future__ import annotations

import numpy as np
from scipy import optimize

from ._richresult import RichResult

__all__ = ["dcc_multivariate_garch"]


def _univariate_garch11(r):
    """Internal GARCH(1,1) for marginal volatilities -- Gaussian MLE."""
    n = r.size
    var_r = float(np.var(r))

    def nll(p):
        w, a, b = p
        if w <= 0 or a < 0 or b < 0 or a + b >= 1:
            return 1e10
        s2 = np.empty(n)
        s2[0] = var_r
        for t in range(1, n):
            s2[t] = w + a * r[t - 1] ** 2 + b * s2[t - 1]
            s2[t] = max(s2[t], 1e-12)
        return 0.5 * np.sum(np.log(2 * np.pi * s2) + r ** 2 / s2)

    fit = optimize.minimize(
        nll, [var_r * 0.05, 0.05, 0.9],
        bounds=[(1e-10, var_r * 10), (1e-8, 0.5), (1e-8, 0.999)],
        method="L-BFGS-B",
    )
    w, a, b = fit.x
    s2 = np.empty(n)
    s2[0] = var_r
    for t in range(1, n):
        s2[t] = w + a * r[t - 1] ** 2 + b * s2[t - 1]
    return s2


def dcc_multivariate_garch(x):
    r"""Engle (2002) Dynamic Conditional Correlation MVGARCH.

    Two-step estimation: per-series GARCH(1,1) for marginals, then
    Gaussian MLE on the standardised residuals to recover :math:`a,b`:

    .. math::

        Q_t = (1-a-b)\,\bar Q + a\,z_{t-1}z_{t-1}' + b\,Q_{t-1},\quad
        R_t = \mathrm{diag}(Q_t)^{-1/2}\,Q_t\,\mathrm{diag}(Q_t)^{-1/2}.

    Parameters
    ----------
    x : array-like, shape (n, k)
        Return panel, n observations × k assets.

    Returns
    -------
    RichResult
        keys: ``a``, ``b``, ``unconditional_correlation``,
        ``conditional_correlation`` (n × k × k), ``conditional_variance``
        (n × k), ``loglik``, ``n``, ``k``, ``method``.

    References
    ----------
    Engle RF (2002). Dynamic Conditional Correlation. *JBES* 20(3),
    339-350.
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    if X.shape[0] < X.shape[1]:
        X = X.T  # assume more rows than columns
    n, k = X.shape
    if n < 30 or k < 2:
        raise ValueError(f"Need n>=30, k>=2; got n={n}, k={k}.")

    # Step 1: GARCH(1,1) marginals on each demeaned series.
    H = np.empty((n, k))
    Z = np.empty((n, k))
    for j in range(k):
        rj = X[:, j] - X[:, j].mean()
        H[:, j] = _univariate_garch11(rj)
        Z[:, j] = rj / np.sqrt(H[:, j] + 1e-12)

    Q_bar = (Z.T @ Z) / n

    def dcc_nll(p):
        a, b = p
        if a < 0 or b < 0 or a + b >= 0.9999:
            return 1e10
        Q = Q_bar.copy()
        ll = 0.0
        for t in range(n):
            d = np.sqrt(np.clip(np.diag(Q), 1e-12, None))
            R = Q / np.outer(d, d)
            try:
                sign, logdet = np.linalg.slogdet(R)
                if sign <= 0:
                    return 1e10
                Rinv = np.linalg.inv(R)
            except np.linalg.LinAlgError:
                return 1e10
            zt = Z[t]
            ll += 0.5 * (logdet + zt @ Rinv @ zt - zt @ zt)
            Q = (1 - a - b) * Q_bar + a * np.outer(zt, zt) + b * Q

        return ll

    fit = optimize.minimize(
        dcc_nll, [0.02, 0.95],
        bounds=[(1e-6, 0.5), (1e-6, 0.999)],
        method="L-BFGS-B",
    )
    a, b = fit.x

    # Re-construct path of Q_t and R_t.
    Q = Q_bar.copy()
    R_path = np.empty((n, k, k))
    for t in range(n):
        d = np.sqrt(np.clip(np.diag(Q), 1e-12, None))
        R_path[t] = Q / np.outer(d, d)
        Q = (1 - a - b) * Q_bar + a * np.outer(Z[t], Z[t]) + b * Q

    return RichResult(payload={
        "a": float(a), "b": float(b),
        "unconditional_correlation": Q_bar,
        "conditional_correlation": R_path,
        "conditional_variance": H,
        "loglik": float(-fit.fun),
        "n": int(n), "k": int(k),
        "method": "DCC(1,1) two-step Gaussian MLE (numpy)",
    })


def cheatsheet():
    return "dccmd: DCC multivariate GARCH (Engle 2002)."
