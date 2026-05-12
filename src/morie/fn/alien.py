# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Alienation index for spatial voting."""

from __future__ import annotations

from ._containers import DescriptiveResult


def alienation_index(voter, candidates) -> DescriptiveResult:
    """Compute alienation scores: distance from voter to each candidate.

    High alienation means no candidate is close to the voter.

    :param voter: Voter ideal point.
    :param candidates: Array of candidate positions.
    :return: DescriptiveResult with alienation scores.

    .. epigraph:: "Tatakae!" -- Eren Yeager, Attack on Titan
    """
    import numpy as np

    voter = np.asarray(voter, dtype=float).ravel()
    cands = np.atleast_2d(np.asarray(candidates, dtype=float))
    dists = np.sqrt(np.sum((cands - voter) ** 2, axis=1))
    min_dist = float(np.min(dists))
    return DescriptiveResult(
        name="alienation_index",
        value=min_dist,
        extra={"distances": dists.tolist(), "min_alienation": min_dist, "mean_alienation": float(np.mean(dists))},
    )


alien = alienation_index


def cheatsheet() -> str:
    return "alienation_index({}) -> Alienation index for spatial voting."
