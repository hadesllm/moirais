# morie.fn — function file (hadesllm/morie)
"""Control function approach to endogeneity.

Two-stage procedure: (1) regress the endogenous variable on
instruments to obtain residuals, (2) include those residuals as
additional controls in the outcome equation.  Under correct
specification, the residual absorbs the endogeneity.

.. math::

    \\text{Stage 1: } X = Z \\gamma + v \\\\
    \\text{Stage 2: } Y = X \\beta + \\hat{v} \\rho + W \\delta + \\varepsilon

References
----------
Rivers, D. & Vuong, Q. H. (1988). Limited information estimators and
    exogeneity tests for simultaneous probit models. *Journal of
    Econometrics*, 39(3), 347--366.
Horowitz, J. L. (2009). *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Chapter 6.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def ctrfn(Y: np.ndarray, X: np.ndarray, Z: np.ndarray, W: np.ndarray | None = None, cdf=None, *, alpha: float = 0.05) -> dict[str, Any]:
    r"""Control function estimator for endogeneity correction.

    Parameters
    ----------
    Y : np.ndarray
        Outcome vector, shape ``(n,)``.
    X : np.ndarray
        Endogenous regressor, shape ``(n,)`` or ``(n, 1)``.
    Z : np.ndarray
        Instruments, shape ``(n, q)`` with ``q >= 1``.
    W : np.ndarray or None
        Exogenous covariates, shape ``(n, r)``.  If *None*, intercept only.
    alpha : float
        Significance level for confidence intervals.

    Returns
    -------
    dict[str, Any]
        ``beta`` (endogenous coefficient), ``rho`` (control function
        coefficient), ``se_beta``, ``se_rho``, ``ci_beta``,
        ``hausman_t`` (t-stat for exogeneity test), ``hausman_p``,
        ``n``, ``method``.
    """
    Y = np.asarray(Y, dtype=float).ravel()
    X = np.asarray(X, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]
    n = len(Y)
    if len(X) != n or Z.shape[0] != n:
        raise ValueError("Y, X, Z must have the same number of rows.")

    ones = np.ones((n, 1))
    if W is not None:
        W = np.asarray(W, dtype=float)
        if W.ndim == 1:
            W = W[:, None]
        Z1 = np.column_stack([Z, W, ones])
        W_aug = np.column_stack([W, ones])
    else:
        Z1 = np.column_stack([Z, ones])
        W_aug = ones

    gamma = np.linalg.lstsq(Z1, X, rcond=None)[0]
    v_hat = X - Z1 @ gamma

    M = np.column_stack([X[:, None], v_hat[:, None], W_aug])
    coef = np.linalg.lstsq(M, Y, rcond=None)[0]
    beta = float(coef[0])
    rho = float(coef[1])

    resid = Y - M @ coef
    sigma2 = float(np.sum(resid ** 2) / max(n - M.shape[1], 1))
    MtM_inv = np.linalg.inv(M.T @ M + 1e-10 * np.eye(M.shape[1]))
    se_all = np.sqrt(sigma2 * np.diag(MtM_inv))
    se_beta = float(se_all[0])
    se_rho = float(se_all[1])

    hausman_t = rho / se_rho if se_rho > 1e-12 else 0.0
    hausman_p = float(2 * (1 - stats.norm.cdf(abs(hausman_t))))

    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    return {
        "beta": beta,
        "rho": rho,
        "se_beta": se_beta,
        "se_rho": se_rho,
        "ci_beta": (beta - z_crit * se_beta, beta + z_crit * se_beta),
        "hausman_t": hausman_t,
        "hausman_p": hausman_p,
        "n": n,
        "method": "ControlFunction",
    }


ctrfn_fn = ctrfn


def cheatsheet() -> str:
    return "ctrfn(Y, X, Z) -> Control function endogeneity correction (Rivers & Vuong 1988)."
