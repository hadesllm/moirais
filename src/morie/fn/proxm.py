# morie.fn — function file (hadesllm/morie)
"""Proximity cost vector for spatial voting."""

from __future__ import annotations

from ._containers import DescriptiveResult


def proximity_cost(voter, candidates, metric: str = "euclidean") -> DescriptiveResult:
    """Compute distance from voter to each candidate.

    :param voter: Voter ideal point.
    :param candidates: Array of candidate positions.
    :param metric: 'euclidean' or 'manhattan'.
    :return: DescriptiveResult with distance vector.

    .. epigraph:: "Plus Ultra!" -- All Might, My Hero Academia
    """
    import numpy as np

    voter = np.asarray(voter, dtype=float).ravel()
    candidates = np.atleast_2d(np.asarray(candidates, dtype=float))
    if metric == "manhattan":
        dists = np.sum(np.abs(candidates - voter), axis=1)
    else:
        dists = np.sqrt(np.sum((candidates - voter) ** 2, axis=1))
    return DescriptiveResult(
        name="proximity_cost",
        value=float(np.min(dists)),
        extra={"distances": dists.tolist(), "metric": metric, "closest": int(np.argmin(dists))},
    )


proxm = proximity_cost


def cheatsheet() -> str:
    return "proximity_cost({}) -> Proximity cost vector for spatial voting."
