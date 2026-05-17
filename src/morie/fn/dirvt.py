# morie.fn -- function file (hadesllm/morie)
"""Directional voting model."""

from __future__ import annotations

from ._containers import DescriptiveResult


def directional_vote(voter_ideal, candidates) -> DescriptiveResult:
    """Directional utility: dot product of voter and candidate directions from origin.

    :param voter_ideal: Voter ideal point.
    :param candidates: Array of candidate positions.
    :return: DescriptiveResult with directional utility scores.

    .. epigraph:: Number rules the universe. -- Pythagoras
    """
    import numpy as np

    v = np.asarray(voter_ideal, dtype=float).ravel()
    cands = np.atleast_2d(np.asarray(candidates, dtype=float))
    scores = cands @ v
    winner = int(np.argmax(scores))
    return DescriptiveResult(
        name="directional_vote",
        value=winner,
        extra={"scores": scores.tolist(), "winner": winner},
    )


dirvt = directional_vote


def cheatsheet() -> str:
    return "directional_vote({}) -> Directional voting model."
