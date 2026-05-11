# morie.fn — function file (hadesllm/morie)
"""Hotelling model of spatial competition."""

from __future__ import annotations

from ._containers import DescriptiveResult


def hotelling_model(n_voters: int = 100, n_candidates: int = 2, seed: int = 42) -> DescriptiveResult:
    """Simulate Hotelling's spatial competition model.

    Voters uniformly distributed on [0,1]; candidates converge to median.

    :param n_voters: Number of voters.
    :param n_candidates: Number of candidates.
    :param seed: Random seed.
    :return: DescriptiveResult with equilibrium positions.

    .. epigraph:: "I'll be back." -- T-800, Terminator
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    voters = rng.uniform(0, 1, n_voters)
    med = float(np.median(voters))
    equilibrium = np.full(n_candidates, med)
    return DescriptiveResult(
        name="hotelling_model",
        value=med,
        extra={
            "equilibrium_positions": equilibrium.tolist(),
            "n_voters": n_voters,
            "n_candidates": n_candidates,
            "voter_median": med,
        },
    )


hotlg = hotelling_model


def cheatsheet() -> str:
    return "hotelling_model({}) -> Hotelling model of spatial competition."
