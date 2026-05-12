# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""ABC-MCMC (Marjoram et al. 2003)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Union

import numpy as np


def abc_mcmc(
    simulator: Callable[[np.ndarray], np.ndarray],
    prior_log_density: Callable[[np.ndarray], float],
    observed_summary: Union[list, np.ndarray],
    init: Union[list, np.ndarray],
    *,
    epsilon: float = 0.1,
    proposal_sd: float = 0.1,
    n_iter: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    ABC-MCMC: Approximate Bayesian Computation with MCMC sampling.

    :param simulator: Function: parameter -> simulated summary stats.
    :param prior_log_density: Log prior density function.
    :param observed_summary: Observed summary statistics.
    :param init: Initial parameter vector.
    :param epsilon: Acceptance threshold.
    :param proposal_sd: Random walk proposal SD.
    :param n_iter: Number of MCMC iterations.
    :param seed: Random seed.
    :return: Dictionary with samples, acceptance_rate.

    References
    ----------
    Marjoram, P., et al. (2003). *PNAS*, 100(26), 15324--15328.
    """
    rng = np.random.default_rng(seed)
    obs = np.asarray(observed_summary, dtype=float)
    theta = np.asarray(init, dtype=float).copy()
    d = len(theta)

    sim_cur = np.asarray(simulator(theta), dtype=float)
    dist_cur = float(np.sqrt(np.sum((sim_cur - obs) ** 2)))

    samples = np.empty((n_iter, d))
    accept = 0

    for i in range(n_iter):
        proposal = theta + proposal_sd * rng.standard_normal(d)
        sim_prop = np.asarray(simulator(proposal), dtype=float)
        dist_prop = float(np.sqrt(np.sum((sim_prop - obs) ** 2)))

        if dist_prop <= epsilon:
            log_alpha = prior_log_density(proposal) - prior_log_density(theta)
            if np.log(rng.uniform()) < log_alpha:
                theta = proposal
                dist_cur = dist_prop
                accept += 1

        samples[i] = theta

    return {
        "samples": samples,
        "acceptance_rate": accept / n_iter,
        "n_iter": n_iter,
        "dim": d,
        "epsilon": epsilon,
    }


abcmc = abc_mcmc


def cheatsheet() -> str:
    return "abc_mcmc({}) -> ABC-MCMC (Marjoram et al. 2003)."
