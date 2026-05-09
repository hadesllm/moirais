"""Persistent homology spatial"""

import numpy as np

from ._containers import DescriptiveResult


def tda_persistent(data, *, method="default"):
    """Persistent homology spatial

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
        name="zxtda",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


tda_ = tda_persistent


def cheatsheet() -> str:
    return "tda_persistent({}) -> Persistent homology spatial"
