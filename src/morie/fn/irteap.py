# morie.fn — function file (hadesllm/morie)
"""EAP theta estimation."""

from __future__ import annotations

import numpy as np
from ._richresult import RichResult


def _icc_2pl(theta: float, a: float, b: float) -> float:
    """2PL item characteristic curve."""
    exp_val = a * (theta - b)
    if exp_val > 500:
        return 1.0
    if exp_val < -500:
        return 0.0
    return 1.0 / (1.0 + np.exp(-exp_val))


def irt_eap_theta(
    responses: np.ndarray,
    item_params: dict,
    *,
    prior_mean: float = 0.0,
    prior_sd: float = 1.0,
    n_quad: int = 61,
) -> dict:
    """Expected A Posteriori theta estimation with normal prior.

    Uses Gauss-Hermite-style quadrature over a grid of theta values.

    Parameters
    ----------
    responses : ndarray
        Binary response vector (k,) for one person.
    item_params : dict
        {item_name: {'a': ..., 'b': ...}}.
    prior_mean : float
        Prior mean for theta (default 0).
    prior_sd : float
        Prior SD for theta (default 1).
    n_quad : int
        Number of quadrature points (default 61).

    Returns
    -------
    dict
        Keys: 'theta', 'se'.

    References
    ----------
    Bock, R. D. & Mislevy, R. J. (1982). Adaptive EAP estimation of
    ability in a microcomputer environment. Applied Psychological
    Measurement, 6(4), 431-444.
    """
    r = np.asarray(responses, dtype=np.float64).ravel()
    params_list = list(item_params.values())
    k = len(params_list)
    if len(r) != k:
        raise ValueError(f"responses length {len(r)} != n_items {k}")

    # Quadrature grid: prior_mean +/- 4*prior_sd
    theta_grid = np.linspace(prior_mean - 4 * prior_sd, prior_mean + 4 * prior_sd, n_quad)

    # Log-likelihood at each quadrature point
    log_lik = np.zeros(n_quad)
    for j in range(k):
        if np.isnan(r[j]):
            continue
        a = params_list[j].get("a", 1.0)
        b = params_list[j].get("b", 0.0)
        for q in range(n_quad):
            p = _icc_2pl(theta_grid[q], a, b)
            p = np.clip(p, 1e-15, 1 - 1e-15)
            log_lik[q] += r[j] * np.log(p) + (1 - r[j]) * np.log(1 - p)

    # Prior (normal)
    log_prior = -0.5 * ((theta_grid - prior_mean) / prior_sd) ** 2

    # Posterior (unnormalized, in log space)
    log_post = log_lik + log_prior
    log_post -= log_post.max()  # numerical stability
    post = np.exp(log_post)
    post /= post.sum()

    # EAP estimate
    theta_hat = float(np.sum(theta_grid * post))
    # Posterior SD
    se = float(np.sqrt(np.sum((theta_grid - theta_hat) ** 2 * post)))

    return RichResult(payload={"theta": theta_hat, "se": se})


def cheatsheet() -> str:
    return "_icc_2pl({}) -> EAP theta estimation."
