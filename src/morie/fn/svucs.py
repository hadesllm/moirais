"""Uncovered set in 2D"""

import numpy as np

from ._containers import DescriptiveResult


def uncovered_set(data, *, method="default"):
    """Uncovered set in 2D

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
        name="svucs",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


unco = uncovered_set


def cheatsheet() -> str:
    return "uncovered_set({}) -> Uncovered set in 2D"
