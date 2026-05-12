# morie.fn -- function file (hadesllm/morie)
"""Case-control semiparametric mixture model."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import optimize, stats

__all__ = ["csmix"]


def csmix(
    y: np.ndarray,
    X: np.ndarray,
    case: np.ndarray,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    r"""
    Fit a semiparametric case-control mixture model.

    In case-control designs, the sampling probability depends on case
    status. The retrospective likelihood uses Bayes' theorem:

    .. math::

        P(X \mid D=1) = \frac{P(D=1 \mid X) f(X)}{P(D=1)}

    The model estimates the logistic regression :math:`\text{logit}\,P(D=1 \mid X) = X^T\beta`
    adjusting for the case-control sampling via profile likelihood over
    the intercept.

    :param y: Outcome (not used directly; case status drives the model).
    :param X: Covariate matrix, shape (n, p).
    :param case: Case indicator (1=case, 0=control), shape (n,).
    :param alpha: Significance level. Default 0.05.
    :return: Dict with ``beta``, ``se``, ``or`` (odds ratios), ``ci_lower``,
        ``ci_upper``, ``prevalence_offset``, ``n``, ``n_cases``.
    :raises ValueError: If arrays are empty.

    References
    ----------
    Kosorok, M.R. (2008). Ch. 13. Springer.
    """
    X = np.asarray(X, dtype=float)
    case = np.asarray(case, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    if n == 0:
        raise ValueError("Arrays must be non-empty.")

    X_aug = np.column_stack([np.ones(n), X])
    d = p + 1

    def neg_loglik(beta):
        from scipy.special import expit
        eta = X_aug @ beta
        prob = expit(eta)
        prob = np.clip(prob, 1e-10, 1 - 1e-10)
        return -float(np.sum(case * np.log(prob) + (1 - case) * np.log(1 - prob)))

    beta0 = np.zeros(d)
    result = optimize.minimize(neg_loglik, beta0, method="BFGS")
    beta_hat = result.x

    eps = 1e-5
    hessian = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            ei, ej = np.zeros(d), np.zeros(d)
            ei[i] = eps
            ej[j] = eps
            hessian[i, j] = (neg_loglik(beta_hat + ei + ej) - neg_loglik(beta_hat + ei - ej) - neg_loglik(beta_hat - ei + ej) + neg_loglik(beta_hat - ei - ej)) / (4 * eps ** 2)

    try:
        var = np.linalg.inv(hessian)
        se = np.sqrt(np.maximum(np.diag(var), 0))
    except np.linalg.LinAlgError:
        se = np.full(d, np.nan)

    se_slopes = se[1:]
    beta_slopes = beta_hat[1:]
    z = stats.norm.ppf(1.0 - alpha / 2.0)

    return {
        "beta": beta_slopes,
        "se": se_slopes,
        "or": np.exp(beta_slopes),
        "ci_lower": beta_slopes - z * se_slopes,
        "ci_upper": beta_slopes + z * se_slopes,
        "prevalence_offset": float(beta_hat[0]),
        "n": n,
        "n_cases": int(np.sum(case)),
    }


def cheatsheet() -> str:
    return "csmix({y, X, case}) -> Semiparametric case-control mixture model."
