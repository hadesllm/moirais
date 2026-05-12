# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Majority agenda rule."""

import numpy as np

from ._containers import SpatialResult


def agmaj(options, setter_ideal, reversion):
    """Majority agenda rule.

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
        name="Majority agenda rule",
        statistic=0.0,
        extra={},
    )


agmaj = agmaj  # alias


def cheatsheet() -> str:
    return "agmaj({}) -> Majority agenda rule."
