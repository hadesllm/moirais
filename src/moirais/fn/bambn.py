# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Burn-in trimming for MCMC chains."""

from __future__ import annotations

from ._containers import DescriptiveResult


def bayesian_burnin_trim(chain, burnin=100) -> DescriptiveResult:
    """Trim burn-in samples from an MCMC chain.

    .. epigraph:: "Valar Morghulis." -- Various, Game of Thrones
    """
    import numpy as np

    chain = np.asarray(chain, dtype=float)
    trimmed = chain[burnin:]
    return DescriptiveResult(
        name="bayesian_burnin_trim",
        value=float(trimmed.shape[0]),
        extra={
            "trimmed_chain": trimmed,
            "burnin": burnin,
            "original_length": chain.shape[0],
            "trimmed_length": trimmed.shape[0],
        },
    )


bambn = bayesian_burnin_trim


def cheatsheet() -> str:
    return "bayesian_burnin_trim({}) -> Burn-in trimming for MCMC chains."
