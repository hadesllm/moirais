# morie.fn -- function file (hadesllm/morie)
"""Hamiltonian Monte Carlo sampler."""

from __future__ import annotations

__all__ = ["hamiltonian_mc", "hmc"]

from collections.abc import Callable
from typing import Any, Union

import numpy as np


def hamiltonian_mc(
    log_target: Callable[[np.ndarray], float],
    grad_log_target: Callable[[np.ndarray], np.ndarray],
    init: Union[list, np.ndarray],
    *,
    epsilon: float = 0.05,
    n_leapfrog: int = 20,
    n_iter: int = 2000,
    mass_matrix: Union[np.ndarray, None] = None,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Hamiltonian Monte Carlo (HMC) sampler.

    Uses leapfrog integration of Hamilton's equations to propose
    distant states with high acceptance probability, reducing
    random-walk behavior relative to standard MH.

    Parameters
    ----------
    log_target : callable
        Log-density of target distribution (unnormalized OK).
    grad_log_target : callable
        Gradient of log-density, returning array of shape (d,).
    init : array-like
        Initial parameter vector (d,).
    epsilon : float
        Leapfrog step size.
    n_leapfrog : int
        Number of leapfrog steps per proposal.
    n_iter : int
        Number of HMC iterations.
    mass_matrix : ndarray or None
        Mass matrix (d, d). Default is identity.
    seed : int
        Random seed.

    Returns
    -------
    dict
        samples : ndarray (n_iter, d)
        acceptance_rate : float
        n_iter : int

    Raises
    ------
    ValueError
        If epsilon or n_leapfrog are non-positive.

    References
    ----------
    Neal, R. M. (2011). MCMC using Hamiltonian dynamics. In *Handbook
    of Markov Chain Monte Carlo*, CRC Press, Ch. 5.
    Duane, S., et al. (1987). Phys. Lett. B, 195(2), 216--222.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0.")
    if n_leapfrog < 1:
        raise ValueError("n_leapfrog must be >= 1.")
    if n_iter < 1:
        raise ValueError("n_iter must be >= 1.")

    rng = np.random.default_rng(seed)
    theta = np.asarray(init, dtype=float).copy()
    d = len(theta)

    if mass_matrix is None:
        M_inv = np.eye(d)
    else:
        M_inv = np.linalg.inv(np.asarray(mass_matrix, dtype=float))

    samples = np.empty((n_iter, d))
    accept = 0

    for i in range(n_iter):
        r = rng.standard_normal(d)
        theta_prop = theta.copy()
        r_prop = r.copy()

        r_prop = r_prop + 0.5 * epsilon * grad_log_target(theta_prop)
        for _ in range(n_leapfrog - 1):
            theta_prop = theta_prop + epsilon * M_inv @ r_prop
            r_prop = r_prop + epsilon * grad_log_target(theta_prop)
        theta_prop = theta_prop + epsilon * M_inv @ r_prop
        r_prop = r_prop + 0.5 * epsilon * grad_log_target(theta_prop)
        r_prop = -r_prop

        H_cur = -log_target(theta) + 0.5 * r @ M_inv @ r
        H_prop = -log_target(theta_prop) + 0.5 * r_prop @ M_inv @ r_prop
        log_alpha = H_cur - H_prop

        if np.log(rng.uniform()) < log_alpha:
            theta = theta_prop
            accept += 1

        samples[i] = theta

    return {
        "samples": samples,
        "acceptance_rate": accept / n_iter,
        "n_iter": n_iter,
        "dim": d,
    }


hmc = hamiltonian_mc


def cheatsheet() -> str:
    return "hamiltonian_mc(log_target, grad, init) -> Hamiltonian Monte Carlo sampler."
