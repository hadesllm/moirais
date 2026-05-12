# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Agenda cycling detection."""

import numpy as np

from ._containers import SpatialResult


def agcyc(options, setter_ideal, reversion):
    """Agenda cycling detection.

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
        name="Agenda cycling detection",
        statistic=0.0,
        extra={},
    )


agcyc = agcyc  # alias


def cheatsheet() -> str:
    return "agcyc({}) -> Agenda cycling detection."
