# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Amendment agenda procedure."""

import numpy as np

from ._containers import SpatialResult


def agame(options, setter_ideal, reversion):
    """Amendment agenda procedure.

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
        name="Amendment agenda procedure",
        statistic=0.0,
        extra={},
    )


agame = agame  # alias


def cheatsheet() -> str:
    return "agame({}) -> Amendment agenda procedure."
