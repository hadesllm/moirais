"""Spatial vote choice — pick closest candidate."""

from __future__ import annotations

from ._containers import DescriptiveResult


def spatial_vote(voter_ideal, candidates) -> DescriptiveResult:
    """Return index of closest candidate to voter ideal point.

    :param voter_ideal: Voter ideal point (scalar or array).
    :param candidates: Array of candidate positions (n_candidates x n_dims).
    :return: DescriptiveResult with winning candidate index.

    .. epigraph:: "Make it so." -- Jean-Luc Picard, Star Trek
    """
    import numpy as np

    voter_ideal = np.asarray(voter_ideal, dtype=float)
    candidates = np.atleast_2d(np.asarray(candidates, dtype=float))
    if voter_ideal.ndim == 0 or (
        voter_ideal.ndim == 1 and candidates.ndim == 2 and candidates.shape[1] != voter_ideal.shape[0]
    ):
        voter_ideal = voter_ideal.ravel()
        candidates = candidates.reshape(-1, voter_ideal.shape[0]) if candidates.size > voter_ideal.size else candidates
    dists = np.sqrt(np.sum((candidates - voter_ideal) ** 2, axis=1))
    winner = int(np.argmin(dists))
    return DescriptiveResult(
        name="spatial_vote",
        value=winner,
        extra={"distances": dists.tolist(), "winner": winner},
    )


spvot = spatial_vote


def cheatsheet() -> str:
    return "spatial_vote({}) -> Spatial vote choice — pick closest candidate."
