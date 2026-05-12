# morie.fn -- function file (hadesllm/morie)
"""Linear probability spatial."""

import numpy as np

from ._containers import SpatialResult


def pslin(voter, candidates, *, beta=1.0):
    """Linear probability spatial.

    Parameters
    ----------
    voter : array_like
        Voter ideal point.
    candidates : array_like
        Candidate position.
    beta : float
        Scale parameter.

    Returns
    -------
    SpatialResult
    """
    ideal = np.asarray(voter, dtype=float)
    pos = np.asarray(candidates, dtype=float)
    stat = -float(np.linalg.norm(ideal - pos))
    return SpatialResult(
        name="Linear Spatial Utility",
        statistic=float(stat),
        extra={"euclidean_distance": -stat},
    )


pslin = pslin  # alias


def cheatsheet() -> str:
    return "pslin({}) -> Linear probability spatial."
