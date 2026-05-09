# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Agenda winner prediction."""

import numpy as np

from ._containers import SpatialResult


def agwin(options, setter_ideal, reversion):
    """Agenda winner prediction.

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
        name="Agenda winner prediction",
        statistic=0.0,
        extra={},
    )


agwin = agwin  # alias


def cheatsheet() -> str:
    return "agwin({}) -> Agenda winner prediction."
