# moirais.fn — function file (hadesllm/moirais)
"""Metropolis-Hastings sampler (general)."""

from __future__ import annotations

__all__ = ["metropolis_hastings", "mhsmp"]

from collections.abc import Callable
from typing import Any, Union

import numpy as np


def metropolis_hastings(
    log_target: Callable[[np.ndarray], float],
    init: Union[list, np.ndarray],
    *,
    proposal_cov: Union[list, np.ndarray, None] = None,
    n_iter: int = 10000,
    burn_in: int = 0,
    thin: int = 1,
    seed: int = 42,
) -> dict[str, Any]:
    """
    General-purpose Metropolis-Hastings MCMC sampler.

    Generates samples from an arbitrary target distribution specified
    by its (unnormalized) log-density using a multivariate normal
    random-walk proposal.

    Parameters
    ----------
    log_target : callable
        Function mapping parameter vector (d,) to log-density (unnormalized OK).
    init : array-like
        Initial parameter vector of dimension d.
    proposal_cov : array-like or None
        Proposal covariance matrix (d, d).  Default is identity.
    n_iter : int
        Total number of iterations (including burn-in).
    burn_in : int
        Number of initial samples to discard.
    thin : int
        Keep every ``thin``-th sample after burn-in.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        samples : ndarray of shape (n_kept, d)
        acceptance_rate : float
        n_iter : int
        dim : int

    Raises
    ------
    ValueError
        If n_iter < 1 or burn_in >= n_iter.

    References
    ----------
    Metropolis, N., et al. (1953). J. Chem. Phys., 21(6), 1087--1092.
    Hastings, W. K. (1970). Biometrika, 57(1), 97--109.
    """
    if n_iter < 1:
        raise ValueError("n_iter must be >= 1.")
    if burn_in >= n_iter:
        raise ValueError("burn_in must be < n_iter.")
    if thin < 1:
        raise ValueError("thin must be >= 1.")

    rng = np.random.default_rng(seed)
    theta = np.asarray(init, dtype=float).copy()
    d = len(theta)

    if proposal_cov is None:
        L = np.eye(d)
    else:
        L = np.linalg.cholesky(np.asarray(proposal_cov, dtype=float))

    all_samples = np.empty((n_iter, d))
    log_p_cur = log_target(theta)
    accept = 0

    for i in range(n_iter):
        proposal = theta + L @ rng.standard_normal(d)
        log_p_prop = log_target(proposal)
        log_alpha = log_p_prop - log_p_cur
        if np.log(rng.uniform()) < log_alpha:
            theta = proposal
            log_p_cur = log_p_prop
            accept += 1
        all_samples[i] = theta

    kept = all_samples[burn_in::thin]

    return {
        "samples": kept,
        "acceptance_rate": accept / n_iter,
        "n_iter": n_iter,
        "dim": d,
    }


mhsmp = metropolis_hastings


def cheatsheet() -> str:
    return "metropolis_hastings(log_target, init) -> Metropolis-Hastings MCMC sampler."
