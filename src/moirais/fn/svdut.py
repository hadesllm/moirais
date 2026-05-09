"""Distance-decay utility wrapper."""

import numpy as np

from ._containers import SpatialResult


def svdut(ideal, pos):
    """Distance-decay utility wrapper.

    Returns
    -------
    SpatialResult
    """

    ideal = np.asarray(ideal, dtype=float)
    pos = np.asarray(pos, dtype=float)
    d = float(np.linalg.norm(ideal - pos))
    stat = 1.0 / (1.0 + d)
    return SpatialResult(
        name="Distance-Decay Utility",
        statistic=float(stat),
        extra={"distance": d},
    )


svdut = svdut  # alias


def cheatsheet() -> str:
    return "svdut({}) -> Distance-decay utility wrapper."
