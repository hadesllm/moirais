# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Reversion point agenda."""

import numpy as np

from ._containers import SpatialResult


def agrev(options, setter_ideal, reversion):
    """Reversion point agenda.

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
        name="Reversion point agenda",
        statistic=0.0,
        extra={},
    )


agrev = agrev  # alias


def cheatsheet() -> str:
    return "agrev({}) -> Reversion point agenda."
