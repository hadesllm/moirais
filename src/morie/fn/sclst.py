# morie.fn -- function file (hadesllm/morie)
"""Semiparametric censored least squares (SCLS)."""

from __future__ import annotations

import numpy as np


def sclst(
    y: np.ndarray,
    X: np.ndarray,
    delta: np.ndarray,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    seed: int = 42,
) -> dict:
    r"""
    Semiparametric censored least squares estimator.

    Estimates the linear regression coefficient :math:`\beta` from
    right-censored data :math:`(T_i, \delta_i, X_i)` where
    :math:`T_i = \min(T_i^*, C_i)` and :math:`\delta_i = 1\{T_i^* \le C_i\}`.

    Uses the iterative Buckley-James (1979) approach: in each iteration,
    censored residuals are replaced by their conditional expectation
    under the Kaplan-Meier estimate of the residual distribution,
    and OLS is re-fit:

    .. math::

        \hat{\beta}^{(m+1)} = \arg\min_{\beta}
        \sum_{i=1}^{n} \bigl[ \tilde{e}_i^{(m)} - X_i (\beta - \hat{\beta}^{(m)}) \bigr]^2

    where :math:`\tilde{e}_i^{(m)}` is the imputed residual from iteration *m*.

    :param y: Observed (possibly censored) times, shape ``(n,)``.
    :param X: Covariate matrix, shape ``(n, p)``. Include intercept if desired.
    :param delta: Censoring indicator, shape ``(n,)``.
        1 = event observed, 0 = right-censored.
    :param max_iter: Maximum iterations. Default 100.
    :param tol: Convergence tolerance on coefficient change. Default 1e-6.
    :param seed: Random seed (reserved). Default 42.
    :return: dict with ``coefficients``, ``se``, ``n_iter``, ``converged``,
        ``n_obs``, ``n_events``.
    :raises ValueError: If dimensions are inconsistent.

    References
    ----------
    Buckley, J. & James, I. (1979). Linear regression with censored data.
        *Biometrika*, 66(3), 429--436.
    Horowitz, J. L. (2009). *Semiparametric and Nonparametric Methods
        in Econometrics*. Springer, Section 6.3.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    delta = np.asarray(delta, dtype=float)
    if y.ndim != 1 or delta.ndim != 1:
        raise ValueError("y and delta must be 1-D.")
    if X.ndim != 2:
        raise ValueError("X must be 2-D.")
    n, p = X.shape
    if n != y.shape[0] or n != delta.shape[0]:
        raise ValueError("y, X, and delta must have the same number of rows.")

    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    converged = False
    n_iter = 0

    for it in range(max_iter):
        resid = y - X @ beta
        order = np.argsort(resid)
        e_sorted = resid[order]
        d_sorted = delta[order]

        n_risk = np.arange(n, 0, -1, dtype=float)
        hazard = d_sorted / n_risk
        surv = np.cumprod(1.0 - hazard)

        imputed = resid.copy()
        for i in range(n):
            if delta[i] == 0:
                ei = resid[i]
                mask = e_sorted > ei
                if not np.any(mask):
                    imputed[i] = ei
                    continue
                idx_above = np.where(mask)[0]
                s_at_e = surv[np.searchsorted(e_sorted, ei, side="right") - 1] if np.searchsorted(e_sorted, ei, side="right") > 0 else 1.0
                if s_at_e <= 0:
                    imputed[i] = ei
                    continue
                vals = e_sorted[idx_above]
                s_vals = surv[idx_above]
                s_prev = np.concatenate([[s_at_e], s_vals[:-1]])
                weights = (s_prev - s_vals) / s_at_e
                cond_mean = np.sum(weights * vals)
                tail_prob = s_vals[-1] / s_at_e
                imputed[i] = cond_mean + tail_prob * vals[-1]

        y_imp = X @ beta + imputed
        beta_new = np.linalg.lstsq(X, y_imp, rcond=None)[0]
        n_iter = it + 1

        if np.max(np.abs(beta_new - beta)) < tol:
            converged = True
            beta = beta_new
            break
        beta = beta_new

    resid_final = y - X @ beta
    uncens = delta == 1
    if np.sum(uncens) > p:
        sigma2 = np.sum(resid_final[uncens] ** 2) / (np.sum(uncens) - p)
        XtX_inv = np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.diag(sigma2 * XtX_inv))
    else:
        se = np.full(p, np.nan)

    return {
        "coefficients": beta,
        "se": se,
        "n_iter": n_iter,
        "converged": converged,
        "n_obs": n,
        "n_events": int(np.sum(delta)),
    }


sclst_fn = sclst


def cheatsheet() -> str:
    return "sclst(y, X, delta) -> Semiparametric censored least squares."
