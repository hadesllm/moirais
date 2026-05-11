"""Targeted Minimum Loss-Based Estimation (TMLE) for ATE.

Implements the canonical TMLE algorithm: fit initial outcome model,
compute clever covariate, fluctuate via logistic submodel, then
plug-in estimate with influence-function-based standard errors.

References
----------
van der Laan, M. J. & Rubin, D. (2006). Targeted maximum likelihood
learning. *International Journal of Biostatistics*, 2(1), Article 11.

van der Laan, M. J. & Rose, S. (2011). *Targeted Learning: Causal
Inference for Observational and Experimental Data*. Springer.

Kosorok, M. R. (2008). *Introduction to Empirical Processes and
Semiparametric Inference*. Springer. Chapter 15.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import special, stats


def tmle(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    *,
    outcome_model: str = "logistic",
    ps_trim: float = 0.01,
    alpha: float = 0.05,
) -> dict[str, Any]:
    r"""Estimate the ATE via Targeted Minimum Loss-Based Estimation.

    Algorithm
    ---------
    1. Fit initial outcome model :math:`\hat{Q}^0(T, X)`.
    2. Estimate propensity score :math:`\hat{g}(X) = P(T=1 \mid X)`.
    3. Compute clever covariate :math:`H(T,X) = T/\hat{g}(X) - (1-T)/(1-\hat{g}(X))`.
    4. Fluctuate :math:`\hat{Q}^0` along :math:`H` via univariate logistic
       submodel to get :math:`\hat{Q}^*`.
    5. ATE = :math:`n^{-1}\sum[\hat{Q}^*(1,X_i) - \hat{Q}^*(0,X_i)]`.
    6. SE via influence function.

    Parameters
    ----------
    Y : np.ndarray
        Binary outcome vector (0/1), shape ``(n,)``.
    T : np.ndarray
        Binary treatment vector (0/1), shape ``(n,)``.
    X : np.ndarray
        Covariate matrix, shape ``(n, p)``.
    outcome_model : str
        Initial outcome model type (``"logistic"`` or ``"linear"``).
    ps_trim : float
        Propensity scores are clipped to ``[ps_trim, 1 - ps_trim]``.
    alpha : float
        Significance level for confidence interval.

    Returns
    -------
    dict[str, Any]
        ``ate``, ``se``, ``ci_lower``, ``ci_upper``, ``epsilon``,
        ``n``, ``method``.
    """
    Y = np.asarray(Y, dtype=float)
    T = np.asarray(T, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = len(Y)

    Xd = np.column_stack([X, np.ones(n)])

    ps = _logistic_fit_predict(Xd, T)
    ps = np.clip(ps, ps_trim, 1.0 - ps_trim)

    if outcome_model == "logistic":
        Q0 = _logistic_fit_predict(np.column_stack([T[:, None], X, np.ones(n)]), Y)
        Q0 = np.clip(Q0, 1e-6, 1.0 - 1e-6)
        Q1 = _logistic_fit_predict(
            np.column_stack([np.ones((n, 1)), X, np.ones(n)]), Y,
            fit_X=np.column_stack([T[:, None], X, np.ones(n)]),
        )
        Q0_ctrl = _logistic_fit_predict(
            np.column_stack([np.zeros((n, 1)), X, np.ones(n)]), Y,
            fit_X=np.column_stack([T[:, None], X, np.ones(n)]),
        )
    else:
        beta = _ols_fit(np.column_stack([T[:, None], X, np.ones(n)]), Y)
        Xmat1 = np.column_stack([np.ones((n, 1)), X, np.ones(n)])
        Xmat0 = np.column_stack([np.zeros((n, 1)), X, np.ones(n)])
        Q0 = np.column_stack([T[:, None], X, np.ones(n)]) @ beta
        Q1 = Xmat1 @ beta
        Q0_ctrl = Xmat0 @ beta

    H1 = T / ps
    H0 = -(1.0 - T) / (1.0 - ps)
    H = H1 + H0

    if outcome_model == "logistic":
        Q0_logit = special.logit(np.clip(Q0, 1e-6, 1 - 1e-6))
        epsilon = _logistic_submodel_fit(Q0_logit, H, Y)
        Q_star = special.expit(special.logit(np.clip(Q1, 1e-6, 1 - 1e-6)) + epsilon / ps)
        Q_star0 = special.expit(special.logit(np.clip(Q0_ctrl, 1e-6, 1 - 1e-6)) - epsilon / (1 - ps))
    else:
        epsilon = _linear_submodel_fit(Q0, H, Y)
        Q_star = Q1 + epsilon / ps
        Q_star0 = Q0_ctrl - epsilon / (1 - ps)

    ate = float(np.mean(Q_star - Q_star0))

    ic = (Q_star - Q_star0) - ate + H * (Y - Q0)
    se = float(np.std(ic, ddof=1) / np.sqrt(n))

    z = stats.norm.ppf(1.0 - alpha / 2.0)
    return {
        "ate": ate,
        "se": se,
        "ci_lower": ate - z * se,
        "ci_upper": ate + z * se,
        "epsilon": float(epsilon),
        "n": n,
        "method": "TMLE",
    }


def _logistic_fit_predict(X, y, *, fit_X=None):
    """Minimal logistic regression via IRLS (no sklearn)."""
    Xf = fit_X if fit_X is not None else X
    beta = np.zeros(Xf.shape[1])
    for _ in range(25):
        p = special.expit(Xf @ beta)
        p = np.clip(p, 1e-8, 1 - 1e-8)
        W = p * (1 - p)
        z = Xf @ beta + (y - p) / W
        try:
            beta = np.linalg.solve(Xf.T @ np.diag(W) @ Xf + 1e-8 * np.eye(Xf.shape[1]), Xf.T @ (W * z))
        except np.linalg.LinAlgError:
            break
    return special.expit(X @ beta)


def _ols_fit(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def _logistic_submodel_fit(offset, H, Y):
    eps = 0.0
    for _ in range(50):
        p = special.expit(offset + eps * H)
        p = np.clip(p, 1e-8, 1 - 1e-8)
        score = np.sum(H * (Y - p))
        info = np.sum(H**2 * p * (1 - p))
        if abs(info) < 1e-12:
            break
        eps += score / info
        if abs(score / max(info, 1e-12)) < 1e-8:
            break
    return eps


def _linear_submodel_fit(offset, H, Y):
    resid = Y - offset
    return float(np.sum(H * resid) / np.sum(H**2))


tmle_fn = tmle


def cheatsheet() -> str:
    return "tmle(Y, T, X) -> TMLE ATE estimator (Kosorok 2008, Ch. 15)."
