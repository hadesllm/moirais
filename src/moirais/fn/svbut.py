"""Bimodal utility combining peaks."""

import numpy as np

from ._containers import SpatialResult


def svbut(ideal, pos, peak2, *, alpha=0.5, bw=1.0):
    """Bimodal utility combining peaks.

    Returns
    -------
    SpatialResult
    """

    ideal = np.asarray(ideal, dtype=float)
    pos = np.asarray(pos, dtype=float)
    peak2 = np.asarray(peak2, dtype=float)
    u1 = float(np.exp(-np.sum((ideal - pos) ** 2) / (2 * bw**2)))
    u2 = float(np.exp(-np.sum((peak2 - pos) ** 2) / (2 * bw**2)))
    stat = alpha * u1 + (1 - alpha) * u2
    return SpatialResult(
        name="Bimodal Utility",
        statistic=float(stat),
        extra={"u1": u1, "u2": u2, "alpha": alpha},
    )


svbut = svbut  # alias


def cheatsheet() -> str:
    return "svbut({}) -> Bimodal utility combining peaks."
