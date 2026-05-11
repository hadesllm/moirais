"""S-learner for heterogeneous treatment effect (CATE) estimation.

The S-learner (Single model learner) fits one outcome model with
treatment as a covariate, then imputes counterfactuals by toggling T.

References
----------
Kunzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019).
Metalearners for estimating heterogeneous treatment effects using
machine learning. *PNAS*, 116(10), 4156-4165.

Nie, X., & Wager, S. (2021). Quasi-oracle estimation of heterogeneous
treatment effects. *Biometrika*, 108(2), 299-319.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit

__all__ = ["slear"]


def slear(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    *,
    model: str = "linear",
    alpha: float = 0.05,
) -> dict[str, Any]:
    r"""Estimate CATE via the S-learner (single-model) meta-learner.

    Fits one model :math:`\hat{\mu}(t, \mathbf{x})` with treatment T
    included as a feature.  Individual treatment effects are:

    .. math::

        \hat{\tau}(x_i) = \hat{\mu}(1, x_i) - \hat{\mu}(0, x_i)

    Parameters
    ----------
    Y : np.ndarray
        Outcome vector, shape ``(n,)``.
    T : np.ndarray
        Binary treatment indicator, shape ``(n,)``.
    X : np.ndarray
        Covariate matrix, shape ``(n, p)``.
    model : str
        ``"linear"`` (OLS) or ``"logistic"`` (IRLS).
    alpha : float
        Significance level for ATE confidence interval.

    Returns
    -------
    dict
        ``cate``: array of per-unit estimates, ``ate``, ``se``,
        ``ci_lower``, ``ci_upper``, ``n``, ``method``.

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

    # Design matrix: intercept | T | X
    XT = np.column_stack([np.ones(n), T, X])
    X1 = np.column_stack([np.ones(n), np.ones(n), X])
    X0 = np.column_stack([np.ones(n), np.zeros(n), X])

    if model == "linear":
        beta = np.linalg.lstsq(XT, Y, rcond=None)[0]
        mu1 = X1 @ beta
        mu0 = X0 @ beta
    elif model == "logistic":
        beta = _irls(XT, Y)
        mu1 = expit(X1 @ beta)
        mu0 = expit(X0 @ beta)
    else:
        raise ValueError(f"Unknown model '{model}'. Choose 'linear' or 'logistic'.")

    cate = mu1 - mu0
    ate = float(np.mean(cate))
    se = float(np.std(cate, ddof=1) / np.sqrt(n))
    from scipy.stats import norm as _norm
    z = _norm.ppf(1.0 - alpha / 2.0)

    return {
        "cate": cate,
        "ate": ate,
        "se": se,
        "ci_lower": ate - z * se,
        "ci_upper": ate + z * se,
        "n": n,
        "method": "S-learner",
    }


def _irls(X: np.ndarray, y: np.ndarray, max_iter: int = 30) -> np.ndarray:
    """Iteratively reweighted least squares for logistic regression."""
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        p = np.clip(expit(X @ beta), 1e-8, 1 - 1e-8)
        W = p * (1 - p)
        XtW = X.T * W
        A = XtW @ X + 1e-8 * np.eye(X.shape[1])
        b = XtW @ (X @ beta + (y - p) / W)
        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break
    return beta


def cheatsheet() -> str:
    return "slear(Y, T, X) -> S-learner CATE (Kunzel et al. 2019, PNAS)."
