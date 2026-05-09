# moirais.fn — function file (hadesllm/moirais)
"""Sequential agenda model."""

import numpy as np

from ._containers import SpatialResult


def agseq(options, setter_ideal, reversion):
    """Sequential agenda model.

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
        name="Sequential agenda model",
        statistic=0.0,
        extra={},
    )


agseq = agseq  # alias


def cheatsheet() -> str:
    return "agseq({}) -> Sequential agenda model."
