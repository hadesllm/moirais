# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bayesian AM MCMC sampler."""

from __future__ import annotations

from ._containers import DescriptiveResult


def bayesian_am_mcmc_sample(Z, n_samples=500) -> DescriptiveResult:
    """Metropolis-Hastings MCMC for augmented model.

    .. epigraph:: "Winter is coming." -- Stark, Game of Thrones
    """
    import numpy as np

    Z = np.asarray(Z, dtype=float)
    n, p = Z.shape if Z.ndim == 2 else (Z.shape[0], 1)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    rng = np.random.default_rng(42)
    chain = np.zeros((n_samples, p))
    current = rng.standard_normal(p)
    accept = 0
    for s in range(n_samples):
        proposal = current + 0.1 * rng.standard_normal(p)
        log_ratio = -0.5 * (np.sum(proposal**2) - np.sum(current**2))
        if np.log(rng.uniform()) < log_ratio:
            current = proposal
            accept += 1
        chain[s] = current
    return DescriptiveResult(
        name="bayesian_am_mcmc_sample",
        value=float(accept / n_samples),
        extra={
            "chain": chain,
            "acceptance_rate": float(accept / n_samples),
            "n_samples": n_samples,
            "n_params": p,
        },
    )


bamcm = bayesian_am_mcmc_sample


def cheatsheet() -> str:
    return "bayesian_am_mcmc_sample({}) -> Bayesian AM MCMC sampler."
