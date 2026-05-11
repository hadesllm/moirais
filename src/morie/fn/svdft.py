"""Directional with intensity term."""

import numpy as np

from ._containers import SpatialResult


def svdft(voter, candidates):
    """Directional with intensity term.

    Returns
    -------
    SpatialResult
    """

    voter = np.asarray(voter, dtype=float)
    candidates = np.asarray(candidates, dtype=float)
    voter_int = float(np.linalg.norm(voter))
    scores = np.array([float(np.dot(voter, c) * np.linalg.norm(c)) for c in candidates])
    choice = int(np.argmax(scores))
    stat = float(scores.max())
    return SpatialResult(
        name="Directional with Intensity",
        statistic=float(stat),
        extra={"chosen": choice, "voter_intensity": voter_int},
    )


svdft = svdft  # alias


def cheatsheet() -> str:
    return "svdft({}) -> Directional with intensity term."
