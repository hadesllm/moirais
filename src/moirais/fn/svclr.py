"""Condorcet loser identification"""

import numpy as np

from ._containers import DescriptiveResult


def condorcet_loser(data, *, method="default"):
    """Condorcet loser identification

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
        name="svclr",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


cond = condorcet_loser


def cheatsheet() -> str:
    return "condorcet_loser({}) -> Condorcet loser identification"
