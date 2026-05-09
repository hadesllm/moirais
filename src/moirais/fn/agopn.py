# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Open agenda model."""

import numpy as np

from ._containers import SpatialResult


def agopn(options, setter_ideal, reversion):
    """Open agenda model.

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
        name="Open agenda model",
        statistic=0.0,
        extra={},
    )


agopn = agopn  # alias


def cheatsheet() -> str:
    return "agopn({}) -> Open agenda model."
