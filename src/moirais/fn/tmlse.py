"""TMLE standard error via influence curve."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

__all__ = ["tmlse"]


def tmlse(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    *,
    ate_estimate: float | None = None,
    ps_trim: float = 0.01,
    alpha: float = 0.05,
) -> dict[str, Any]:
    r"""
    Compute TMLE standard error using the efficient influence curve.

    The influence curve for the ATE under TMLE is:

    .. math::

        D^*(O) = H(T,X)(Y - \hat{Q}^*(T,X)) + \hat{Q}^*(1,X) - \hat{Q}^*(0,X) - \psi

    The variance is :math:`\text{Var}(D^*) / n`.

    :param Y: Outcome vector, shape (n,).
    :param T: Binary treatment vector, shape (n,).
    :param X: Covariate matrix, shape (n, p).
    :param ate_estimate: Pre-computed ATE estimate. If None, computed.
    :param ps_trim: Propensity score clip bounds. Default 0.01.
    :param alpha: Significance level. Default 0.05.
    :return: Dict with ``se``, ``ci_lower``, ``ci_upper``, ``influence_values``,
        ``ate``, ``n``.

    References
    ----------
    Kosorok, M.R. (2008). Ch. 16. Springer.
    van der Laan & Rose (2011). *Targeted Learning*. Springer.
    """
    Y = np.asarray(Y, dtype=float)
    T = np.asarray(T, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = len(Y)

    Xd = np.column_stack([X, np.ones(n)])
    ps = _logistic_predict(Xd, T)
    ps = np.clip(ps, ps_trim, 1.0 - ps_trim)

    Xo = np.column_stack([T[:, None], X, np.ones(n)])
    beta = np.linalg.lstsq(Xo, Y, rcond=None)[0]
    mu1 = np.column_stack([np.ones((n, 1)), X, np.ones(n)]) @ beta
    mu0 = np.column_stack([np.zeros((n, 1)), X, np.ones(n)]) @ beta
    mu_obs = np.where(T == 1, mu1, mu0)

    H = T / ps - (1 - T) / (1 - ps)

    if ate_estimate is None:
        ate_estimate = float(np.mean(mu1 - mu0))

    ic = H * (Y - mu_obs) + mu1 - mu0 - ate_estimate
    se = float(np.std(ic, ddof=1) / np.sqrt(n))
    z = stats.norm.ppf(1.0 - alpha / 2.0)

    return {
        "se": se,
        "ci_lower": ate_estimate - z * se,
        "ci_upper": ate_estimate + z * se,
        "influence_values": ic,
        "ate": ate_estimate,
        "n": n,
    }


def _logistic_predict(X, y):
    from scipy.special import expit
    beta = np.zeros(X.shape[1])
    for _ in range(25):
        p = expit(X @ beta)
        p = np.clip(p, 1e-8, 1 - 1e-8)
        W = p * (1 - p)
        z = X @ beta + (y - p) / W
        try:
            beta = np.linalg.solve(X.T @ np.diag(W) @ X + 1e-8 * np.eye(X.shape[1]), X.T @ (W * z))
        except np.linalg.LinAlgError:
            break
    return expit(X @ beta)


def cheatsheet() -> str:
    return "tmlse(Y, T, X) -> TMLE standard error via influence curve."
