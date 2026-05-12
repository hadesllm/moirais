# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Agenda game tree analysis."""

import numpy as np

from ._containers import SpatialResult


def aggta(options, setter_ideal, reversion):
    """Agenda game tree analysis.

    Returns
    -------
    SpatialResult
    """

    options = np.asarray(options, dtype=float)
    setter_ideal = float(setter_ideal)
    reversion = float(reversion)
    sq_dist = abs(setter_ideal - reversion)
    best_opt = options[np.argmin(np.abs(options - setter_ideal))]
    stat = abs(best_opt - setter_ideal) < sq_dist
    return SpatialResult(
        name="Agenda game tree analysis",
        statistic=0.0,
        extra={},
    )


aggta = aggta  # alias


def cheatsheet() -> str:
    return "aggta({}) -> Agenda game tree analysis."
