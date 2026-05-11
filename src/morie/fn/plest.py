# morie.fn — function file (hadesllm/morie)
"""Profile likelihood estimation."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import optimize, stats

__all__ = ["plest"]


def plest(
    x: np.ndarray,
    log_likelihood: callable,
    *,
    n_params: int = 2,
    theta0: np.ndarray | None = None,
    alpha: float = 0.05,
    grid_points: int = 100,
) -> dict[str, Any]:
    r"""
    Profile likelihood estimation and confidence intervals.

    The profile likelihood for the parameter of interest :math:`\psi` is:

    .. math::

        \ell_p(\psi) = \max_\eta \ell(\psi, \eta)

    Profile likelihood CI: :math:`\{\psi : 2[\ell_p(\hat\psi) - \ell_p(\psi)] \le \chi^2_{1,\alpha}\}`.

    :param x: Observation array, shape (n,) or (n, p).
    :param log_likelihood: Function(x, theta) -> scalar log-likelihood.
    :param n_params: Total number of parameters. Default 2.
    :param theta0: Initial parameter vector. Default zeros.
    :param alpha: Significance level. Default 0.05.
    :param grid_points: Grid size for profile. Default 100.
    :return: Dict with ``mle``, ``profile_ci``, ``psi_grid``, ``profile_ll``,
        ``max_ll``.
    :raises ValueError: If x is empty.

    References
    ----------
    Kosorok, M.R. (2008). Ch. 14. Springer.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("x must be non-empty.")

    if theta0 is None:
        theta0 = np.zeros(n_params)
    theta0 = np.asarray(theta0, dtype=float)

    def neg_ll(theta):
        return -log_likelihood(x, theta)

    result = optimize.minimize(neg_ll, theta0, method="Nelder-Mead")
    mle = result.x
    max_ll = -result.fun

    psi_hat = mle[0]
    se_approx = 1.0

    eps = 1e-5
    h1 = (neg_ll(mle + np.array([eps] + [0] * (n_params - 1))) + neg_ll(mle - np.array([eps] + [0] * (n_params - 1))) - 2 * neg_ll(mle)) / eps ** 2
    if h1 > 0:
        se_approx = 1.0 / np.sqrt(h1)

    psi_range = np.linspace(psi_hat - 4 * se_approx, psi_hat + 4 * se_approx, grid_points)
    profile_ll = np.zeros(grid_points)

    for k, psi_val in enumerate(psi_range):
        if n_params > 1:
            def neg_ll_fixed(eta, pv=psi_val):
                theta = np.concatenate([[pv], eta])
                return -log_likelihood(x, theta)
            res = optimize.minimize(neg_ll_fixed, mle[1:], method="Nelder-Mead")
            profile_ll[k] = -res.fun
        else:
            profile_ll[k] = log_likelihood(x, np.array([psi_val]))

    chi2_crit = stats.chi2.ppf(1.0 - alpha, df=1)
    in_ci = 2 * (max_ll - profile_ll) <= chi2_crit
    if np.any(in_ci):
        ci_lower = float(psi_range[in_ci][0])
        ci_upper = float(psi_range[in_ci][-1])
    else:
        ci_lower = float(psi_hat)
        ci_upper = float(psi_hat)

    return {
        "mle": mle,
        "profile_ci": (ci_lower, ci_upper),
        "psi_grid": psi_range,
        "profile_ll": profile_ll,
        "max_ll": float(max_ll),
    }


def cheatsheet() -> str:
    return "plest({x, log_likelihood}) -> Profile likelihood estimation."
