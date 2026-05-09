"""Bliss point estimation"""

import numpy as np

from ._containers import DescriptiveResult


def bliss_point(data, *, method="default"):
    """Bliss point estimation

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
        name="svblp",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


blis = bliss_point


def cheatsheet() -> str:
    return "bliss_point({}) -> Bliss point estimation"
