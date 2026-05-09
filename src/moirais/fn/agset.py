# moirais.fn — function file (hadesllm/moirais)
"""Agenda setter model."""

import numpy as np

from ._containers import SpatialResult


def agset(options, setter_ideal, reversion):
    """Agenda setter model.

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
        name="Agenda setter model",
        statistic=0.0,
        extra={},
    )


agset = agset  # alias


def cheatsheet() -> str:
    return "agset({}) -> Agenda setter model."
