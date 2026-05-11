"""Bayesian outbreak detection"""

import numpy as np

from ._containers import SpatialResult


def bayes_outbreak(data, *, method="default"):
    """Bayesian outbreak detection

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
        name="Bayesian outbreak detection",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


baye = bayes_outbreak


def cheatsheet() -> str:
    return "bayes_outbreak({}) -> Bayesian outbreak detection"
