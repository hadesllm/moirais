"""Substitution estimator (G-computation / plug-in)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

__all__ = ["submd"]


def submd(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    r"""
    Compute the substitution (plug-in / G-computation) estimator of the ATE.

    .. math::

        \hat\psi_{\text{sub}} = \frac{1}{n}\sum_{i=1}^n
        [\hat{Q}(1, X_i) - \hat{Q}(0, X_i)]

    where :math:`\hat{Q}(t, x) = E[Y \mid T=t, X=x]` is the outcome model.

    :param Y: Outcome vector, shape (n,).
    :param T: Binary treatment vector, shape (n,).
    :param X: Covariate matrix, shape (n, p).
    :param alpha: Significance level. Default 0.05.
    :return: Dict with ``ate``, ``se``, ``ci_lower``, ``ci_upper``,
        ``mu1_mean``, ``mu0_mean``, ``n``, ``method``.

    References
    ----------
    Kosorok, M.R. (2008). Ch. 16-17. Springer.
    """
    Y = np.asarray(Y, dtype=float)
    T = np.asarray(T, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = len(Y)

    Xo = np.column_stack([T[:, None], X, np.ones(n)])
    beta = np.linalg.lstsq(Xo, Y, rcond=None)[0]

    mu1 = np.column_stack([np.ones((n, 1)), X, np.ones(n)]) @ beta
    mu0 = np.column_stack([np.zeros((n, 1)), X, np.ones(n)]) @ beta

    ate = float(np.mean(mu1 - mu0))
    ic = mu1 - mu0 - ate
    se = float(np.std(ic, ddof=1) / np.sqrt(n))
    z = stats.norm.ppf(1.0 - alpha / 2.0)

    return {
        "ate": ate,
        "se": se,
        "ci_lower": ate - z * se,
        "ci_upper": ate + z * se,
        "mu1_mean": float(np.mean(mu1)),
        "mu0_mean": float(np.mean(mu0)),
        "n": n,
        "method": "Substitution",
    }


def cheatsheet() -> str:
    return "submd(Y, T, X) -> Substitution (G-computation) ATE estimator."
