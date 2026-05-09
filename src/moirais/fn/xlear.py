"""X-learner for heterogeneous treatment effect (CATE) estimation.

The X-learner (Cross learner) builds on the T-learner by using
imputed counterfactual residuals and propensity-score-weighted
combination for improved efficiency in unbalanced designs.

References
----------
Kunzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019).
Metalearners for estimating heterogeneous treatment effects using
machine learning. *PNAS*, 116(10), 4156-4165.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit
from scipy.stats import norm as _norm

__all__ = ["xlear"]


def xlear(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    *,
    model: str = "linear",
    alpha: float = 0.05,
) -> dict[str, Any]:
    r"""Estimate CATE via the X-learner meta-learner.

    Steps
    -----
    1. Fit separate T-learner outcome models
       :math:`\hat{\mu}_1, \hat{\mu}_0`.
    2. Impute pseudo-outcomes:

       .. math::

           D^1_i = Y_i - \hat{\mu}_0(X_i), \quad T_i = 1

           D^0_i = \hat{\mu}_1(X_i) - Y_i, \quad T_i = 0

    3. Regress :math:`D^1` on :math:`X` (treated) and :math:`D^0` on
       :math:`X` (control) to get :math:`\hat{\tau}_1, \hat{\tau}_0`.
    4. Combine via propensity score:

       .. math::

           \hat{\tau}(x) = g(x)\hat{\tau}_0(x) + (1-g(x))\hat{\tau}_1(x)

    Parameters
    ----------
    Y : np.ndarray
        Outcome vector, shape ``(n,)``.
    T : np.ndarray
        Binary treatment indicator, shape ``(n,)``.
    X : np.ndarray
        Covariate matrix, shape ``(n, p)``.
    model : str
        ``"linear"`` for OLS base learners.
    alpha : float
        Significance level for ATE CI.

    Returns
    -------
    dict
        ``cate``, ``ate``, ``se``, ``ci_lower``, ``ci_upper``,
        ``n``, ``method``.

    References
    ----------
    Kunzel et al. (2019). PNAS, 116(10), 4156-4165.
    """
    Y = np.asarray(Y, dtype=float)
    T = np.asarray(T, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = len(Y)
    if not (len(T) == n == X.shape[0]):
        raise ValueError("Y, T, X must have the same number of observations.")

    idx1 = T == 1
    idx0 = T == 0
    if idx1.sum() < 2 or idx0.sum() < 2:
        raise ValueError("Both treatment groups need at least 2 observations.")

    Xa = np.column_stack([np.ones(n), X])

    # Stage 1: T-learner outcome models
    beta1 = np.linalg.lstsq(Xa[idx1], Y[idx1], rcond=None)[0]
    beta0 = np.linalg.lstsq(Xa[idx0], Y[idx0], rcond=None)[0]
    mu1 = Xa @ beta1
    mu0 = Xa @ beta0

    # Stage 2: imputed pseudo-outcomes
    D1 = Y[idx1] - mu0[idx1]          # treated pseudo-residuals
    D0 = mu1[idx0] - Y[idx0]          # control pseudo-residuals

    # Stage 3: cross-models
    gamma1 = np.linalg.lstsq(Xa[idx1], D1, rcond=None)[0]
    gamma0 = np.linalg.lstsq(Xa[idx0], D0, rcond=None)[0]
    tau1 = Xa @ gamma1
    tau0 = Xa @ gamma0

    # Stage 4: propensity-weighted combination
    ps = np.clip(_irls_ps(Xa, T), 1e-6, 1 - 1e-6)
    cate = ps * tau0 + (1.0 - ps) * tau1

    ate = float(np.mean(cate))
    se = float(np.std(cate, ddof=1) / np.sqrt(n))
    z = _norm.ppf(1.0 - alpha / 2.0)

    return {
        "cate": cate,
        "ate": ate,
        "se": se,
        "ci_lower": ate - z * se,
        "ci_upper": ate + z * se,
        "n": n,
        "method": "X-learner",
    }


def _irls_ps(X: np.ndarray, y: np.ndarray, max_iter: int = 25) -> np.ndarray:
    """Logistic propensity score via IRLS."""
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        p = np.clip(expit(X @ beta), 1e-8, 1 - 1e-8)
        W = p * (1 - p)
        A = (X.T * W) @ X + 1e-8 * np.eye(X.shape[1])
        b = (X.T * W) @ (X @ beta + (y - p) / W)
        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break
    return expit(X @ beta)


def cheatsheet() -> str:
    return "xlear(Y, T, X) -> X-learner CATE (Kunzel et al. 2019, PNAS)."
