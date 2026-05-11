# morie.fn — function file (hadesllm/morie)
"""Empirical Bayes (parametric)."""

from __future__ import annotations

from typing import Any, Union

import numpy as np
from scipy.optimize import minimize_scalar


def empirical_bayes(
    estimates: Union[list, np.ndarray],
    standard_errors: Union[list, np.ndarray],
) -> dict[str, Any]:
    """
    Parametric empirical Bayes shrinkage (normal-normal model).

    Estimates the prior variance tau^2 from data, then computes
    posterior means (shrinkage estimates).

    :param estimates: Observed estimates (k,).
    :param standard_errors: Standard errors of estimates (k,).
    :return: Dictionary with shrunk_estimates, tau2, shrinkage_factors.

    References
    ----------
    Morris, C. N. (1983). *JASA*, 78(381), 47--55.
    Efron, B. (2010). *Large-Scale Inference*, Cambridge University Press.
    """
    theta_hat = np.asarray(estimates, dtype=float).ravel()
    se = np.asarray(standard_errors, dtype=float).ravel()
    k = len(theta_hat)
    sigma2 = se ** 2

    def neg_marginal_ll(log_tau2):
        tau2 = np.exp(log_tau2)
        V = sigma2 + tau2
        ll = -0.5 * np.sum(np.log(V) + theta_hat ** 2 / V)
        return -ll

    res = minimize_scalar(neg_marginal_ll, bounds=(-10, 20), method="bounded")
    tau2 = np.exp(res.x)

    B = sigma2 / (sigma2 + tau2)
    shrunk = (1 - B) * theta_hat

    return {
        "shrunk_estimates": shrunk.tolist(),
        "tau2": float(tau2),
        "shrinkage_factors": B.tolist(),
        "grand_mean": 0.0,
        "k": k,
    }


empby = empirical_bayes


def cheatsheet() -> str:
    return "empirical_bayes({}) -> Empirical Bayes (parametric)."
