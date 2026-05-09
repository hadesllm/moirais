"""Gaussian utility function for spatial voting."""

import numpy as np

from ._containers import SpatialResult


def svgut(ideal, pos, *, bandwidth=1.0):
    """Gaussian utility function for spatial voting.

    Returns
    -------
    SpatialResult
    """

    ideal = np.asarray(ideal, dtype=float)
    pos = np.asarray(pos, dtype=float)
    d2 = float(np.sum((ideal - pos) ** 2))
    stat = np.exp(-d2 / (2 * bandwidth**2))
    return SpatialResult(
        name="Gaussian utility function for spatial voting",
        statistic=0.0,
        extra={},
    )


svgut = svgut  # alias


def cheatsheet() -> str:
    return "svgut({}) -> Gaussian utility function for spatial voting."
