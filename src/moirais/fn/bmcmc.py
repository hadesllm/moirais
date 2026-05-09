# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Metropolis-Hastings MCMC sampler."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def metropolis_hastings(
    log_target: callable,
    initial: float,
    *,
    n_iter: int = 10000,
    proposal_sd: float = 1.0,
    seed: int | None = None,
) -> DescriptiveResult:
    """
    Simple Metropolis-Hastings sampler for a univariate target.

    Parameters
    ----------
    log_target : callable
        Function returning log of unnormalised target density.
    initial : float
        Starting value.
    n_iter : int
        Number of iterations.
    proposal_sd : float
        Standard deviation of Gaussian proposal.
    seed : int, optional
        Random seed.

    Returns
    -------
    DescriptiveResult
        extra has 'samples', 'acceptance_rate'.

    References
    ----------
    Metropolis, N., et al. (1953). Equation of state calculations by
    fast computing machines. *J Chem Phys*, 21(6), 1087-1092.
    """
    rng = np.random.default_rng(seed)
    samples = np.zeros(n_iter)
    samples[0] = initial
    accepted = 0
    current_ll = log_target(initial)

    for i in range(1, n_iter):
        proposal = samples[i - 1] + rng.normal(0, proposal_sd)
        proposal_ll = log_target(proposal)
        log_ratio = proposal_ll - current_ll
        if np.log(rng.uniform()) < log_ratio:
            samples[i] = proposal
            current_ll = proposal_ll
            accepted += 1
        else:
            samples[i] = samples[i - 1]

    rate = accepted / (n_iter - 1)

    return DescriptiveResult(
        name="metropolis_hastings",
        value=float(np.mean(samples[n_iter // 2 :])),
        extra={
            "samples": samples,
            "acceptance_rate": float(rate),
            "n_iter": n_iter,
        },
    )


bmcmc = metropolis_hastings


def cheatsheet() -> str:
    return "metropolis_hastings({}) -> Metropolis-Hastings MCMC sampler."
