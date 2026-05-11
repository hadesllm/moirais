# morie.fn — function file (hadesllm/morie)
"""Hotelling linear competition."""

import numpy as np

from ._containers import SpatialResult


def hslin(party_a, party_b, *, median_voter=0.0):
    """Hotelling linear competition.

    Returns
    -------
    SpatialResult
    """

    ideal = np.asarray(party_a, dtype=float)
    pos = np.asarray(party_b, dtype=float)
    stat = -float(np.linalg.norm(ideal - pos))
    return SpatialResult(
        name="Linear Spatial Utility",
        statistic=float(stat),
        extra={"euclidean_distance": -stat},
    )


hslin = hslin  # alias


def cheatsheet() -> str:
    return "hslin({}) -> Hotelling linear competition."
