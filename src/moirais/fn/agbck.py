# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Backward induction agenda."""

import numpy as np

from ._containers import SpatialResult


def agbck(options, setter_ideal, reversion):
    """Backward induction agenda.

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
        name="Backward induction agenda",
        statistic=0.0,
        extra={},
    )


agbck = agbck  # alias


def cheatsheet() -> str:
    return "agbck({}) -> Backward induction agenda."
