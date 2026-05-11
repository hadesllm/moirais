"""Possibilistic spatial clustering"""

import numpy as np

from ._containers import DescriptiveResult


def possibilistic_sp(data, *, method="default"):
    """Possibilistic spatial clustering

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
        name="zxpos",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


poss = possibilistic_sp


def cheatsheet() -> str:
    return "possibilistic_sp({}) -> Possibilistic spatial clustering"
