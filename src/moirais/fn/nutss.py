# moirais.fn — function file (hadesllm/moirais)
"""Simplified No-U-Turn Sampler (NUTS)."""

from __future__ import annotations

__all__ = ["nuts_sampler", "nutss"]

from collections.abc import Callable
from typing import Any, Union

import numpy as np


def _leapfrog(theta, r, grad_fn, epsilon):
    """Single leapfrog step."""
    r_half = r + 0.5 * epsilon * grad_fn(theta)
    theta_new = theta + epsilon * r_half
    r_new = r_half + 0.5 * epsilon * grad_fn(theta_new)
    return theta_new, r_new


def _build_tree(theta, r, u, v, j, epsilon, log_target, grad_fn, rng):
    """Recursively build a balanced binary tree for NUTS."""
    if j == 0:
        theta_p, r_p = _leapfrog(theta, r, grad_fn, v * epsilon)
        H = log_target(theta_p) - 0.5 * np.dot(r_p, r_p)
        n_p = 1 if np.log(u) <= H else 0
        s_p = 1 if np.log(u) < H + 1000.0 else 0
        return theta_p, r_p, theta_p, r_p, theta_p, n_p, s_p

    (theta_m, r_m, theta_p, r_p, theta_prime,
     n_prime, s_prime) = _build_tree(
        theta, r, u, v, j - 1, epsilon, log_target, grad_fn, rng
    )

    if s_prime == 1:
        if v == -1:
            (theta_m, r_m, _, _, theta_dbl,
             n_dbl, s_dbl) = _build_tree(
                theta_m, r_m, u, v, j - 1, epsilon, log_target, grad_fn, rng
            )
        else:
            (_, _, theta_p, r_p, theta_dbl,
             n_dbl, s_dbl) = _build_tree(
                theta_p, r_p, u, v, j - 1, epsilon, log_target, grad_fn, rng
            )

        total = n_prime + n_dbl
        if total > 0 and rng.uniform() < n_dbl / total:
            theta_prime = theta_dbl
        n_prime = total

        delta = theta_p - theta_m
        s_prime = s_dbl * int(np.dot(delta, r_m) >= 0) * int(np.dot(delta, r_p) >= 0)

    return theta_m, r_m, theta_p, r_p, theta_prime, n_prime, s_prime


def nuts_sampler(
    log_target: Callable[[np.ndarray], float],
    grad_log_target: Callable[[np.ndarray], np.ndarray],
    init: Union[list, np.ndarray],
    *,
    epsilon: float = 0.1,
    n_iter: int = 1000,
    max_tree_depth: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Simplified No-U-Turn Sampler (NUTS).

    Adaptively selects the number of leapfrog steps per iteration
    using the U-turn criterion, avoiding the manual tuning of
    trajectory length required by standard HMC.

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
    n_iter : int
        Number of samples to draw.
    max_tree_depth : int
        Maximum binary tree depth (limits trajectory to 2^depth steps).
    seed : int
        Random seed.

    Returns
    -------
    dict
        samples : ndarray (n_iter, d)
        mean_tree_depth : float
        n_iter : int

    References
    ----------
    Hoffman, M. D. & Gelman, A. (2014). The No-U-Turn Sampler.
    *JMLR*, 15, 1593--1623.
    """
    if n_iter < 1:
        raise ValueError("n_iter must be >= 1.")
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0.")

    rng = np.random.default_rng(seed)
    theta = np.asarray(init, dtype=float).copy()
    d = len(theta)
    samples = np.empty((n_iter, d))
    depths = np.empty(n_iter, dtype=int)

    for i in range(n_iter):
        r0 = rng.standard_normal(d)
        H0 = log_target(theta) - 0.5 * np.dot(r0, r0)
        u = rng.uniform(0, np.exp(H0))

        theta_m = theta.copy()
        theta_p = theta.copy()
        r_m = r0.copy()
        r_p = r0.copy()
        theta_prime = theta.copy()
        j = 0
        n = 1
        s = 1

        while s == 1 and j < max_tree_depth:
            v = 2 * int(rng.uniform() < 0.5) - 1
            if v == -1:
                (theta_m, r_m, _, _, theta_cand,
                 n_cand, s_cand) = _build_tree(
                    theta_m, r_m, u, v, j, epsilon, log_target,
                    grad_log_target, rng
                )
            else:
                (_, _, theta_p, r_p, theta_cand,
                 n_cand, s_cand) = _build_tree(
                    theta_p, r_p, u, v, j, epsilon, log_target,
                    grad_log_target, rng
                )

            if s_cand == 1 and n_cand > 0 and rng.uniform() < min(1.0, n_cand / n):
                theta_prime = theta_cand
            n += n_cand
            delta = theta_p - theta_m
            s = s_cand * int(np.dot(delta, r_m) >= 0) * int(np.dot(delta, r_p) >= 0)
            j += 1

        theta = theta_prime
        samples[i] = theta
        depths[i] = j

    return {
        "samples": samples,
        "mean_tree_depth": float(np.mean(depths)),
        "n_iter": n_iter,
        "dim": d,
    }


nutss = nuts_sampler


def cheatsheet() -> str:
    return "nuts_sampler(log_target, grad, init) -> No-U-Turn Sampler (NUTS)."
