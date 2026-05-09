# moirais.fn — function file (hadesllm/moirais)
"""Score-based efficient estimation."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

__all__ = ["sceff"]


def sceff(
    x: np.ndarray,
    score_func: callable,
    *,
    theta_init: float = 0.0,
    max_iter: int = 50,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> dict[str, Any]:
    r"""
    Efficient score-based estimation via iterative Newton-Raphson.

    Solves :math:`\sum_{i=1}^n S_{\text{eff}}(X_i, \theta) = 0`
    where :math:`S_{\text{eff}}` is the efficient score function.

    :param x: Observation array, shape (n,).
    :param score_func: Efficient score function S(x_i, theta) -> scalar.
    :param theta_init: Initial parameter value. Default 0.
    :param max_iter: Maximum Newton iterations. Default 50.
    :param tol: Convergence tolerance. Default 1e-8.
    :param alpha: Significance level. Default 0.05.
    :return: Dict with ``theta``, ``se``, ``ci_lower``, ``ci_upper``,
        ``n_iter``, ``converged``, ``n``.
    :raises ValueError: If x is empty.

    References
    ----------
    Kosorok, M.R. (2008). Ch. 10. Springer.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("x must be non-empty.")

    n = x.size
    theta = theta_init
    eps = 1e-6
    converged = False

    for it in range(max_iter):
        scores = np.array([score_func(x[i], theta) for i in range(n)])
        total_score = np.sum(scores)

        scores_plus = np.array([score_func(x[i], theta + eps) for i in range(n)])
        deriv = np.sum(scores_plus - scores) / eps

        if abs(deriv) < 1e-12:
            break

        step = total_score / (-deriv)
        theta = theta + step

        if abs(step) < tol:
            converged = True
            break

    scores_final = np.array([score_func(x[i], theta) for i in range(n)])
    info = float(np.mean(scores_final ** 2))
    se = float(1.0 / np.sqrt(n * max(info, 1e-12)))

    z = stats.norm.ppf(1.0 - alpha / 2.0)
    return {
        "theta": float(theta),
        "se": se,
        "ci_lower": float(theta) - z * se,
        "ci_upper": float(theta) + z * se,
        "n_iter": it + 1,
        "converged": converged,
        "n": n,
    }


def cheatsheet() -> str:
    return "sceff({x, score_func}) -> Score-based efficient estimation."
