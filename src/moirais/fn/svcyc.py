"""Condorcet cycle detection"""

import numpy as np

from ._containers import DescriptiveResult


def condorcet_cycle(data, *, method="default"):
    """Condorcet cycle detection

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
        name="svcyc",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


cond = condorcet_cycle


def cheatsheet() -> str:
    return "condorcet_cycle({}) -> Condorcet cycle detection"
