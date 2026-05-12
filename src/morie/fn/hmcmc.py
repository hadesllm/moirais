# morie.fn -- function file (hadesllm/morie)
"""Hamiltonian Monte Carlo sampler."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Union

import numpy as np


def hamiltonian_mc(
    log_target: Callable[[np.ndarray], float],
    grad_log_target: Callable[[np.ndarray], np.ndarray],
    init: Union[list, np.ndarray],
    *,
    step_size: float = 0.01,
    n_leapfrog: int = 20,
    n_iter: int = 5000,
    mass: Union[np.ndarray, None] = None,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Hamiltonian Monte Carlo (HMC) sampler.

    :param log_target: Log-density of target (unnormalized OK).
    :param grad_log_target: Gradient of log-density.
    :param init: Initial parameter vector (d,).
    :param step_size: Leapfrog step size epsilon.
    :param n_leapfrog: Number of leapfrog steps L.
    :param n_iter: Number of HMC iterations.
    :param mass: Mass matrix (d, d). Default identity.
    :param seed: Random seed.
    :return: Dictionary with samples, acceptance_rate.

    References
    ----------
    Neal, R. M. (2011). *Handbook of Markov Chain Monte Carlo*, Ch. 5.
    """
    rng = np.random.default_rng(seed)
    theta = np.asarray(init, dtype=float).copy()
    d = len(theta)

    if mass is None:
        mass_inv = np.eye(d)
    else:
        mass_inv = np.linalg.inv(np.asarray(mass, dtype=float))

    samples = np.empty((n_iter, d))
    accept = 0

    for i in range(n_iter):
        p = rng.standard_normal(d)
        q = theta.copy()
        p_cur = p.copy()

        p = p + 0.5 * step_size * grad_log_target(q)
        for _ in range(n_leapfrog - 1):
            q = q + step_size * mass_inv @ p
            p = p + step_size * grad_log_target(q)
        q = q + step_size * mass_inv @ p
        p = p + 0.5 * step_size * grad_log_target(q)
        p = -p

        current_H = -log_target(theta) + 0.5 * float(p_cur @ mass_inv @ p_cur)
        proposed_H = -log_target(q) + 0.5 * float(p @ mass_inv @ p)

        if np.log(rng.uniform()) < current_H - proposed_H:
            theta = q
            accept += 1

        samples[i] = theta

    return {
        "samples": samples,
        "acceptance_rate": accept / n_iter,
        "n_iter": n_iter,
        "dim": d,
    }


hmcmc = hamiltonian_mc


def cheatsheet() -> str:
    return "hamiltonian_mc({}) -> Hamiltonian Monte Carlo sampler."
