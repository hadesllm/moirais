# moirais.fn — function file (hadesllm/moirais)
"""Nuisance parameter profiling."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import optimize

__all__ = ["nusnc"]


def nusnc(
    x: np.ndarray,
    log_likelihood: callable,
    *,
    n_interest: int = 1,
    theta0: np.ndarray | None = None,
    grid_points: int = 50,
) -> dict[str, Any]:
    r"""
    Profile out nuisance parameters from a likelihood.

    For parameter vector :math:`\theta = (\psi, \eta)` where :math:`\psi` is
    the parameter of interest and :math:`\eta` is the nuisance parameter,
    the profile likelihood is:

    .. math::

        \ell_p(\psi) = \max_\eta \ell(\psi, \eta)

    :param x: Observation array, shape (n,) or (n, p).
    :param log_likelihood: Function(x, theta) -> scalar log-likelihood.
    :param n_interest: Number of parameters of interest (first n_interest). Default 1.
    :param theta0: Initial full parameter vector. Default zeros of length n_interest+1.
    :param grid_points: Grid size for profiling. Default 50.
    :return: Dict with ``psi_grid``, ``profile_ll``, ``psi_mle``, ``nuisance_mle``,
        ``max_ll``.
    :raises ValueError: If x is empty.

    References
    ----------
    Kosorok, M.R. (2008). Ch. 9. Springer.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("x must be non-empty.")

    if theta0 is None:
        theta0 = np.zeros(n_interest + 1)
    theta0 = np.asarray(theta0, dtype=float)
    d = theta0.size
    n_nuisance = d - n_interest

    def neg_ll(theta):
        return -log_likelihood(x, theta)

    result = optimize.minimize(neg_ll, theta0, method="Nelder-Mead")
    theta_mle = result.x
    psi_mle = theta_mle[:n_interest]
    nuisance_mle = theta_mle[n_interest:]
    max_ll = -result.fun

    psi_range = np.linspace(psi_mle[0] - 3.0, psi_mle[0] + 3.0, grid_points)
    profile_ll = np.zeros(grid_points)

    for k, psi_val in enumerate(psi_range):
        if n_nuisance > 0:
            def neg_ll_fixed(eta, psi_v=psi_val):
                theta = np.concatenate([[psi_v], eta])
                return -log_likelihood(x, theta)
            res = optimize.minimize(neg_ll_fixed, nuisance_mle, method="Nelder-Mead")
            profile_ll[k] = -res.fun
        else:
            profile_ll[k] = log_likelihood(x, np.array([psi_val]))

    return {
        "psi_grid": psi_range,
        "profile_ll": profile_ll,
        "psi_mle": psi_mle,
        "nuisance_mle": nuisance_mle,
        "max_ll": max_ll,
    }


def cheatsheet() -> str:
    return "nusnc({x, log_likelihood}) -> Nuisance parameter profiling."
