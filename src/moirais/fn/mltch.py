# moirais.fn — function file (hadesllm/moirais)
"""Combine multiple MCMC chains."""

from __future__ import annotations

from ._containers import DescriptiveResult


def multiple_chains_combine(chains) -> DescriptiveResult:
    """Stack multiple MCMC chains into one combined matrix.

    .. epigraph:: "Gold shall be their crowns." -- Prophecy, Game of Thrones
    """
    import numpy as np

    arrs = [np.asarray(c, dtype=float) for c in chains]
    combined = np.vstack(arrs) if arrs[0].ndim >= 2 else np.concatenate(arrs)
    return DescriptiveResult(
        name="multiple_chains_combine",
        value=float(combined.shape[0]),
        extra={
            "combined": combined,
            "n_chains": len(chains),
            "total_samples": combined.shape[0],
            "per_chain": [len(c) for c in arrs],
        },
    )


mltch = multiple_chains_combine


def cheatsheet() -> str:
    return "multiple_chains_combine({}) -> Combine multiple MCMC chains."
