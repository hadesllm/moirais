"""Three-candidate spatial equilibrium"""

import numpy as np

from ._containers import DescriptiveResult


def hotelling_3cand(data, *, method="default"):
    """Three-candidate spatial equilibrium

    Returns
    -------
    DescriptiveResult
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    mu = float(np.mean(data))
    var = float(np.var(data, ddof=1)) if n > 1 else 0.0
    se = float(np.sqrt(var / n)) if n > 0 else 0.0
    return DescriptiveResult(
        name="svht3",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


hote = hotelling_3cand


def cheatsheet() -> str:
    return "hotelling_3cand({}) -> Three-candidate spatial equilibrium"
