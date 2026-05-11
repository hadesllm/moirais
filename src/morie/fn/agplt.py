# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Platform agenda comparison."""

import numpy as np

from ._containers import SpatialResult


def agplt(options, setter_ideal, reversion):
    """Platform agenda comparison.

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
        name="Platform agenda comparison",
        statistic=0.0,
        extra={},
    )


agplt = agplt  # alias


def cheatsheet() -> str:
    return "agplt({}) -> Platform agenda comparison."
