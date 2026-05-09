# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Restricted agenda model."""

import numpy as np

from ._containers import SpatialResult


def agrst(options, setter_ideal, reversion):
    """Restricted agenda model.

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
        name="Restricted agenda model",
        statistic=0.0,
        extra={},
    )


agrst = agrst  # alias


def cheatsheet() -> str:
    return "agrst({}) -> Restricted agenda model."
