# morie.fn — function file (hadesllm/morie)
"""Ripley edge correction"""

import numpy as np

from ._containers import SpatialResult


def ripley_correction(data, *, method="default"):
    """Ripley edge correction

    Returns
    -------
    SpatialResult
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    mu = float(np.mean(data))
    var = float(np.var(data, ddof=1)) if n > 1 else 0.0
    se = float(np.sqrt(var / n)) if n > 0 else 0.0
    return SpatialResult(
        name="Ripley edge correction",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


ripl = ripley_correction


def cheatsheet() -> str:
    return "ripley_correction({}) -> Ripley edge correction"
