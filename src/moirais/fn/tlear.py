"""T-learner for heterogeneous treatment effect (CATE) estimation.

The T-learner (Two-model learner) fits separate outcome models for the
treated and control groups, then contrasts the two predictions.

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

__all__ = ["tlear"]


def tlear(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    *,
    model: str = "linear",
    alpha: float = 0.05,
) -> dict[str, Any]:
    r"""Estimate CATE via the T-learner (two-model) meta-learner.

    Fits separate models :math:`\hat{\mu}_1(\mathbf{x})` on treated
    observations and :math:`\hat{\mu}_0(\mathbf{x})` on control
    observations.  Individual effects are:

    .. math::

        \hat{\tau}(x_i) = \hat{\mu}_1(x_i) - \hat{\mu}_0(x_i)

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
        raise ValueError("Both treatment groups must have at least 2 observations.")

    X1_fit = np.column_stack([np.ones(idx1.sum()), X[idx1]])
    X0_fit = np.column_stack([np.ones(idx0.sum()), X[idx0]])
    X_all = np.column_stack([np.ones(n), X])

    if model == "linear":
        beta1 = np.linalg.lstsq(X1_fit, Y[idx1], rcond=None)[0]
        beta0 = np.linalg.lstsq(X0_fit, Y[idx0], rcond=None)[0]
        mu1 = X_all @ beta1
        mu0 = X_all @ beta0
    elif model == "logistic":
        beta1 = _irls(X1_fit, Y[idx1])
        beta0 = _irls(X0_fit, Y[idx0])
        mu1 = expit(X_all @ beta1)
        mu0 = expit(X_all @ beta0)
    else:
        raise ValueError(f"Unknown model '{model}'. Use 'linear' or 'logistic'.")

    cate = mu1 - mu0
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
        "method": "T-learner",
    }


def _irls(X: np.ndarray, y: np.ndarray, max_iter: int = 30) -> np.ndarray:
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
    return "tlear(Y, T, X) -> T-learner CATE (Kunzel et al. 2019, PNAS)."
