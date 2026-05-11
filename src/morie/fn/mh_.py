# morie.fn — function file (hadesllm/morie)
"""Metropolis-Hastings MCMC sampler."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def metropolis_hastings(
    log_posterior: Callable[[float], float],
    *,
    start: float = 0.0,
    n_iter: int = 10000,
    proposal_sd: float = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Random-walk Metropolis-Hastings sampler for a univariate target.

    At each iteration, propose theta* ~ N(theta_current, proposal_sd^2).
    Accept with probability min(1, exp(log_posterior(theta*) - log_posterior(theta_current))).

    :param log_posterior: Callable returning the log (unnormalised)
        posterior density at a given parameter value.
    :param start: Starting value of the chain.
    :param n_iter: Number of iterations (default 10000).
    :param proposal_sd: Standard deviation of the Gaussian proposal.
    :param seed: Random seed for reproducibility.
    :return: Dictionary with samples (array), acceptance_rate (float),
        log_posterior_trace (array).
    :raises ValueError: If n_iter < 1 or proposal_sd <= 0.

    References
    ----------
    Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H.,
    & Teller, E. (1953). Equation of state calculations by fast computing
    machines. *Journal of Chemical Physics*, 21(6), 1087--1092.

    Hastings, W. K. (1970). Monte Carlo sampling methods using Markov
    chains and their applications. *Biometrika*, 57(1), 97--109.
    """
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1.")
    if proposal_sd <= 0:
        raise ValueError("proposal_sd must be positive.")

    rng = np.random.default_rng(seed)
    samples = np.empty(n_iter)
    lp_trace = np.empty(n_iter)
    current = float(start)
    current_lp = float(log_posterior(current))
    accepted = 0

    for i in range(n_iter):
        proposal = current + rng.normal(0, proposal_sd)
        proposal_lp = float(log_posterior(proposal))
        log_alpha = proposal_lp - current_lp

        if np.log(rng.uniform()) < log_alpha:
            current = proposal
            current_lp = proposal_lp
            accepted += 1

        samples[i] = current
        lp_trace[i] = current_lp

    return {
        "samples": samples,
        "acceptance_rate": accepted / n_iter,
        "log_posterior_trace": lp_trace,
    }


mh_ = metropolis_hastings


def cheatsheet() -> str:
    return "metropolis_hastings({}) -> Metropolis-Hastings MCMC sampler."
