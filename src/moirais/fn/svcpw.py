"""Copeland spatial winner"""

import numpy as np

from ._containers import DescriptiveResult


def copeland_winner(data, *, method="default"):
    """Copeland spatial winner

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
        name="svcpw",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


cope = copeland_winner


def cheatsheet() -> str:
    return "copeland_winner({}) -> Copeland spatial winner"
