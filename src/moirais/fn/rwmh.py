# moirais.fn — function file (hadesllm/moirais)
"""Random walk Metropolis-Hastings with adaptive step size."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Union

import numpy as np


def random_walk_mh(
    log_target: Callable[[np.ndarray], float],
    init: Union[list, np.ndarray],
    *,
    step_size: float = 1.0,
    n_iter: int = 10000,
    adapt_interval: int = 100,
    target_accept: float = 0.234,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Random walk Metropolis-Hastings with optional step-size adaptation.

    Adapts step_size during first half of iterations to target acceptance rate.

    :param log_target: Log-density of target (unnormalized OK).
    :param init: Initial parameter vector (d,).
    :param step_size: Initial proposal standard deviation.
    :param n_iter: Number of iterations.
    :param adapt_interval: Adaptation window size.
    :param target_accept: Target acceptance rate (0.234 optimal for d>5).
    :param seed: Random seed.
    :return: Dictionary with samples, acceptance_rate, final_step_size.

    References
    ----------
    Roberts, G. O., et al. (1997). *Annals of Applied Probability*, 7(1).
    """
    rng = np.random.default_rng(seed)
    theta = np.asarray(init, dtype=float).copy()
    d = len(theta)

    samples = np.empty((n_iter, d))
    log_p_cur = log_target(theta)
    accept = 0
    window_accept = 0

    for i in range(n_iter):
        proposal = theta + step_size * rng.standard_normal(d)
        log_p_prop = log_target(proposal)
        log_alpha = log_p_prop - log_p_cur
        if np.log(rng.uniform()) < log_alpha:
            theta = proposal
            log_p_cur = log_p_prop
            accept += 1
            window_accept += 1
        samples[i] = theta

        if i < n_iter // 2 and (i + 1) % adapt_interval == 0:
            rate = window_accept / adapt_interval
            if rate > target_accept:
                step_size *= 1.1
            else:
                step_size *= 0.9
            window_accept = 0

    return {
        "samples": samples,
        "acceptance_rate": accept / n_iter,
        "final_step_size": float(step_size),
        "n_iter": n_iter,
        "dim": d,
    }


rwmh = random_walk_mh


def cheatsheet() -> str:
    return "random_walk_mh({}) -> Random walk Metropolis-Hastings with adaptive step size."
