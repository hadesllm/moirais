# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""MCMC chain thinning."""

from __future__ import annotations

from ._containers import DescriptiveResult


def bayesian_thinning(chain, thin=5) -> DescriptiveResult:
    """Thin an MCMC chain by keeping every nth sample.

    .. epigraph:: "Valar Dohaeris." -- Various, Game of Thrones
    """
    import numpy as np

    chain = np.asarray(chain, dtype=float)
    thinned = chain[::thin]
    return DescriptiveResult(
        name="bayesian_thinning",
        value=float(thinned.shape[0]),
        extra={
            "thinned_chain": thinned,
            "thin": thin,
            "original_length": chain.shape[0],
            "thinned_length": thinned.shape[0],
        },
    )


bamth = bayesian_thinning


def cheatsheet() -> str:
    return "bayesian_thinning({}) -> MCMC chain thinning."
