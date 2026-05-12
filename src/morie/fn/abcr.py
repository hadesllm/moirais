# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Approximate Bayesian computation (rejection)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Union

import numpy as np


def abc_rejection(
    simulator: Callable[[np.ndarray], np.ndarray],
    prior_sampler: Callable[[np.random.Generator], np.ndarray],
    observed_summary: Union[list, np.ndarray],
    *,
    epsilon: float = 0.1,
    n_particles: int = 1000,
    n_accepted: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """
    ABC rejection sampler.

    :param simulator: Function that takes parameter vector and returns simulated summary statistics.
    :param prior_sampler: Function that takes rng and returns a parameter draw from the prior.
    :param observed_summary: Observed summary statistics.
    :param epsilon: Acceptance threshold.
    :param n_particles: Maximum number of proposals.
    :param n_accepted: Target number of accepted particles.
    :param seed: Random seed.
    :return: Dictionary with accepted_params, acceptance_rate, distances.

    References
    ----------
    Beaumont, M. A., et al. (2002). *Genetics*, 162(4), 2025--2035.
    """
    rng = np.random.default_rng(seed)
    obs = np.asarray(observed_summary, dtype=float)

    accepted_params = []
    distances = []
    total = 0

    while len(accepted_params) < n_accepted and total < n_particles:
        theta = prior_sampler(rng)
        sim = simulator(theta)
        sim = np.asarray(sim, dtype=float)
        dist = float(np.sqrt(np.sum((sim - obs) ** 2)))
        total += 1
        if dist <= epsilon:
            accepted_params.append(theta.tolist() if hasattr(theta, 'tolist') else [float(theta)])
            distances.append(dist)

    return {
        "accepted_params": accepted_params,
        "n_accepted": len(accepted_params),
        "n_total": total,
        "acceptance_rate": len(accepted_params) / total if total > 0 else 0.0,
        "distances": distances,
        "epsilon": epsilon,
    }


abcr = abc_rejection


def cheatsheet() -> str:
    return "abc_rejection({}) -> Approximate Bayesian computation (rejection)."
