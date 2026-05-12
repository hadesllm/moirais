# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Binary agenda tree."""

import numpy as np

from ._containers import SpatialResult


def agbll(options, setter_ideal, reversion):
    """Binary agenda tree.

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
        name="Binary agenda tree",
        statistic=0.0,
        extra={},
    )


agbll = agbll  # alias


def cheatsheet() -> str:
    return "agbll({}) -> Binary agenda tree."
