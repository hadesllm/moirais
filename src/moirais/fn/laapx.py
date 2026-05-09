# moirais.fn — function file (hadesllm/moirais)
"""Laplace approximation for posterior."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Union

import numpy as np
from scipy.optimize import minimize


def laplace_approximation(
    log_posterior: Callable[[np.ndarray], float],
    init: Union[list, np.ndarray],
    *,
    method: str = "L-BFGS-B",
) -> dict[str, Any]:
    """
    Laplace approximation: approximate posterior as Gaussian at the MAP.

    :param log_posterior: Log-posterior (unnormalized OK).
    :param init: Initial parameter vector (d,).
    :param method: Optimization method for scipy.optimize.minimize.
    :return: Dictionary with MAP estimate, covariance, log_marginal_approx.

    References
    ----------
    Tierney, L. & Kadane, J. (1986). *JASA*, 81(393), 82--86.
    """
    x0 = np.asarray(init, dtype=float)
    d = len(x0)

    res = minimize(lambda x: -log_posterior(x), x0, method=method)
    mode = res.x
    log_post_mode = -res.fun

    eps = 1e-5
    H = np.zeros((d, d))
    f0 = log_posterior(mode)
    for i in range(d):
        for j in range(i, d):
            e_i = np.zeros(d)
            e_j = np.zeros(d)
            e_i[i] = eps
            e_j[j] = eps
            fpp = log_posterior(mode + e_i + e_j)
            fpm = log_posterior(mode + e_i - e_j)
            fmp = log_posterior(mode - e_i + e_j)
            fmm = log_posterior(mode - e_i - e_j)
            H[i, j] = (fpp - fpm - fmp + fmm) / (4 * eps * eps)
            H[j, i] = H[i, j]

    neg_H = -H
    try:
        cov = np.linalg.inv(neg_H)
        sign, log_det = np.linalg.slogdet(cov)
        log_marginal = log_post_mode + 0.5 * d * np.log(2 * np.pi) + 0.5 * log_det
    except np.linalg.LinAlgError:
        cov = np.full((d, d), float("nan"))
        log_marginal = float("nan")

    return {
        "mode": mode.tolist(),
        "covariance": cov.tolist(),
        "se": np.sqrt(np.abs(np.diag(cov))).tolist(),
        "log_marginal_approx": float(log_marginal),
        "converged": bool(res.success),
    }


laapx = laplace_approximation


def cheatsheet() -> str:
    return "laplace_approximation({}) -> Laplace approximation for posterior."
