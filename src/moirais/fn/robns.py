# moirais.fn — function file (hadesllm/moirais)
"""Robinson double-residual estimator with kernel choice.

Estimates the parametric component :math:`\\theta` in the partially
linear model

.. math::

    Y_i = X_i^\\top \\theta + g(Z_i) + \\varepsilon_i

by first nonparametrically removing the effect of :math:`Z` from both
:math:`Y` and :math:`X` via kernel regression, then running OLS on the
residuals.

References
----------
Robinson, P. M. (1988). Root-N-consistent semiparametric regression.
    *Econometrica*, 56(4), 931--954.
Horowitz, J. L. (2009). *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Chapter 3.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def robns(
    Y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    *,
    kernel: str = "gaussian",
    bandwidth: float | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    r"""Robinson double-residual semiparametric estimator.

    Parameters
    ----------
    Y : np.ndarray
        Outcome vector, shape ``(n,)``.
    X : np.ndarray
        Linear covariates, shape ``(n, p)``.
    Z : np.ndarray
        Nonparametric covariates, shape ``(n,)`` or ``(n, q)``.
    kernel : str
        Kernel type: ``"gaussian"`` or ``"epanechnikov"``.
    bandwidth : float or None
        Kernel bandwidth.  If *None*, uses Silverman's rule on *Z*.
    alpha : float
        Significance level.

    Returns
    -------
    dict[str, Any]
        ``theta`` (coefficients), ``se``, ``ci_lower``, ``ci_upper``,
        ``sigma2``, ``n``, ``p``, ``method``.
    """
    Y = np.asarray(Y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if Z.ndim == 1:
        Z = Z[:, None]
    n, p = X.shape
    if len(Y) != n or Z.shape[0] != n:
        raise ValueError("Y, X, Z must have the same number of rows.")

    if bandwidth is None:
        bandwidth = 1.06 * np.std(Z[:, 0]) * n ** (-1.0 / 5.0)
    if bandwidth <= 0:
        raise ValueError(f"bandwidth must be > 0, got {bandwidth}.")

    def _kernel_weights(z_i: np.ndarray) -> np.ndarray:
        diff = Z - z_i[None, :]
        u = np.linalg.norm(diff, axis=1) / bandwidth
        if kernel == "epanechnikov":
            w = np.where(u <= 1, 0.75 * (1 - u ** 2), 0.0)
        else:
            w = np.exp(-0.5 * u ** 2)
        s = w.sum()
        return w / s if s > 0 else np.ones(n) / n

    Y_hat = np.empty(n)
    X_hat = np.empty((n, p))
    for i in range(n):
        w = _kernel_weights(Z[i])
        Y_hat[i] = w @ Y
        X_hat[i] = w @ X

    Y_tilde = Y - Y_hat
    X_tilde = X - X_hat

    theta, residuals, _, _ = np.linalg.lstsq(X_tilde, Y_tilde, rcond=None)
    eps = Y_tilde - X_tilde @ theta
    sigma2 = float(np.sum(eps ** 2) / max(n - p, 1))

    XtX_inv = np.linalg.inv(X_tilde.T @ X_tilde + 1e-10 * np.eye(p))
    se = np.sqrt(sigma2 * np.diag(XtX_inv))

    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    return {
        "theta": theta,
        "se": se,
        "ci_lower": theta - z_crit * se,
        "ci_upper": theta + z_crit * se,
        "sigma2": sigma2,
        "n": n,
        "p": p,
        "method": "Robinson",
    }


robns_fn = robns


def cheatsheet() -> str:
    return "robns(Y, X, Z) -> Robinson double-residual estimator (Robinson 1988)."
