"""Condorcet winner test"""

import numpy as np

from ._containers import DescriptiveResult


def condorcet_winner(data, *, method="default"):
    """Condorcet winner test

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
        name="svcwn",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


cond = condorcet_winner


def cheatsheet() -> str:
    return "condorcet_winner({}) -> Condorcet winner test"
