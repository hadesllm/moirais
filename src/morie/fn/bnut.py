# morie.fn -- function file (hadesllm/morie)
"""No-U-Turn sampler (simplified NUTS)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Union

import numpy as np


def nuts_sampler(
    log_target: Callable[[np.ndarray], float],
    grad_log_target: Callable[[np.ndarray], np.ndarray],
    init: Union[list, np.ndarray],
    *,
    step_size: float = 0.01,
    max_depth: int = 10,
    n_iter: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Simplified No-U-Turn Sampler (NUTS).

    Automatically tunes the number of leapfrog steps using the U-turn criterion.

    :param log_target: Log-density of target (unnormalized OK).
    :param grad_log_target: Gradient of log-density.
    :param init: Initial parameter vector (d,).
    :param step_size: Leapfrog step size epsilon.
    :param max_depth: Maximum tree depth.
    :param n_iter: Number of NUTS iterations.
    :param seed: Random seed.
    :return: Dictionary with samples, acceptance_rate, mean_tree_depth.

    References
    ----------
    Hoffman, M. D. & Gelman, A. (2014). *JMLR*, 15, 1593--1623.
    """
    rng = np.random.default_rng(seed)
    theta = np.asarray(init, dtype=float).copy()
    d = len(theta)

    samples = np.empty((n_iter, d))
    accept = 0
    total_depth = 0

    def _leapfrog(q, p, eps):
        p_new = p + 0.5 * eps * grad_log_target(q)
        q_new = q + eps * p_new
        p_new = p_new + 0.5 * eps * grad_log_target(q_new)
        return q_new, p_new

    def _hamiltonian(q, p):
        return log_target(q) - 0.5 * float(p @ p)

    def _build_tree(q, p, u, v, j, eps):
        if j == 0:
            q_new, p_new = _leapfrog(q, p, v * eps)
            h = _hamiltonian(q_new, p_new)
            n_valid = 1 if np.log(u) <= h else 0
            s = 1 if np.log(u) < h + 1000 else 0
            return q_new, p_new, q_new, p_new, q_new, n_valid, s, 1
        else:
            qm, pm, qp, pp, q_prime, n_prime, s_prime, depth = _build_tree(q, p, u, v, j - 1, eps)
            if s_prime == 1:
                if v == -1:
                    qm, pm, _, _, q_dbl, n_dbl, s_dbl, d2 = _build_tree(qm, pm, u, v, j - 1, eps)
                else:
                    _, _, qp, pp, q_dbl, n_dbl, s_dbl, d2 = _build_tree(qp, pp, u, v, j - 1, eps)
                if n_prime + n_dbl > 0 and rng.uniform() < n_dbl / (n_prime + n_dbl):
                    q_prime = q_dbl
                uturn = float((qp - qm) @ pm) >= 0 and float((qp - qm) @ pp) >= 0
                s_prime = s_dbl * (1 if uturn else 0)
                n_prime = n_prime + n_dbl
                depth = max(depth, d2)
            return qm, pm, qp, pp, q_prime, n_prime, s_prime, depth

    for i in range(n_iter):
        p0 = rng.standard_normal(d)
        u = rng.uniform(0, np.exp(_hamiltonian(theta, p0)))
        qm = theta.copy()
        qp = theta.copy()
        pm = p0.copy()
        pp = p0.copy()
        j = 0
        q_next = theta.copy()
        n = 1
        s = 1

        while s == 1 and j < max_depth:
            v = 2 * int(rng.uniform() > 0.5) - 1
            if v == -1:
                qm, pm, _, _, q_prime, n_prime, s_prime, depth = _build_tree(qm, pm, u, v, j, step_size)
            else:
                _, _, qp, pp, q_prime, n_prime, s_prime, depth = _build_tree(qp, pp, u, v, j, step_size)
            if s_prime == 1 and n_prime > 0 and rng.uniform() < min(1, n_prime / n):
                q_next = q_prime
            uturn = float((qp - qm) @ pm) >= 0 and float((qp - qm) @ pp) >= 0
            s = s_prime * (1 if uturn else 0)
            n = n + n_prime
            j += 1

        if not np.array_equal(q_next, theta):
            accept += 1
        total_depth += j
        theta = q_next
        samples[i] = theta

    return {
        "samples": samples,
        "acceptance_rate": accept / n_iter,
        "mean_tree_depth": total_depth / n_iter,
        "n_iter": n_iter,
        "dim": d,
    }


bnut = nuts_sampler


def cheatsheet() -> str:
    return "nuts_sampler({}) -> No-U-Turn sampler (simplified NUTS)."
