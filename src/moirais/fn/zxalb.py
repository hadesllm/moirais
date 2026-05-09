"""Albers equal-area projection"""

import numpy as np

from ._containers import DescriptiveResult


def albers_proj(data, *, method="default"):
    """Albers equal-area projection

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
        name="zxalb",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


albe = albers_proj


def cheatsheet() -> str:
    return "albers_proj({}) -> Albers equal-area projection"
