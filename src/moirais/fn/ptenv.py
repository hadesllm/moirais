# moirais.fn — function file (hadesllm/moirais)
"""Point pattern Monte Carlo envelope"""

import numpy as np

from ._containers import SpatialResult


def pp_envelope(data, *, method="default"):
    """Point pattern Monte Carlo envelope

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
        name="Point pattern Monte Carlo envelope",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


pp_e = pp_envelope


def cheatsheet() -> str:
    return "pp_envelope({}) -> Point pattern Monte Carlo envelope"
