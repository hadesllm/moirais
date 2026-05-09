# moirais.fn — function file (hadesllm/moirais)
"""Semiparametric efficiency bound (information lower bound)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

__all__ = ["sefnl"]


def sefnl(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    *,
    functional: str = "ate",
    ps_trim: float = 0.01,
    alpha: float = 0.05,
) -> dict[str, Any]:
    r"""
    Compute the semiparametric efficiency bound for the ATE.

    The efficiency bound for :math:`\psi = E[Y(1)] - E[Y(0)]` is:

    .. math::

        \mathcal{I}^{-1} = E\!\left[\frac{\text{Var}(Y \mid T=1, X)}{e(X)}
        + \frac{\text{Var}(Y \mid T=0, X)}{1 - e(X)}
        + (\mu_1(X) - \mu_0(X) - \psi)^2\right]

    :param Y: Outcome vector, shape (n,).
    :param T: Binary treatment vector, shape (n,).
    :param X: Covariate matrix, shape (n, p).
    :param functional: ``"ate"`` (default).
    :param ps_trim: Propensity score clip bounds. Default 0.01.
    :param alpha: Significance level. Default 0.05.
    :return: Dict with ``efficiency_bound``, ``min_se``, ``min_ci_width``,
        ``n``, ``functional``.
    :raises ValueError: If arrays are empty or mismatched.

    References
    ----------
    Kosorok, M.R. (2008). Ch. 10. Springer.
    """
    Y = np.asarray(Y, dtype=float)
    T = np.asarray(T, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = len(Y)
    if n == 0:
        raise ValueError("Arrays must be non-empty.")

    Xd = np.column_stack([X, np.ones(n)])
    ps = _logistic_predict(Xd, T)
    ps = np.clip(ps, ps_trim, 1.0 - ps_trim)

    Xo = np.column_stack([T[:, None], X, np.ones(n)])
    beta = np.linalg.lstsq(Xo, Y, rcond=None)[0]
    mu1 = np.column_stack([np.ones((n, 1)), X, np.ones(n)]) @ beta
    mu0 = np.column_stack([np.zeros((n, 1)), X, np.ones(n)]) @ beta

    resid = Y - np.where(T == 1, mu1, mu0)
    var_y1 = np.zeros(n)
    var_y0 = np.zeros(n)
    for i in range(n):
        if T[i] == 1:
            var_y1[i] = resid[i] ** 2
            var_y0[i] = np.mean(resid[T == 0] ** 2)
        else:
            var_y0[i] = resid[i] ** 2
            var_y1[i] = np.mean(resid[T == 1] ** 2)

    psi = float(np.mean(mu1 - mu0))
    bound = float(np.mean(var_y1 / ps + var_y0 / (1 - ps) + (mu1 - mu0 - psi) ** 2))
    min_se = float(np.sqrt(bound / n))
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    min_ci_width = 2 * z * min_se

    return {
        "efficiency_bound": bound,
        "min_se": min_se,
        "min_ci_width": min_ci_width,
        "n": n,
        "functional": functional,
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
    return "sefnl(Y, T, X) -> Semiparametric efficiency bound."
