"""Party sorting index (Levendusky)"""

import numpy as np

from ._containers import DescriptiveResult


def party_sorting(data, *, method="default"):
    """Party sorting index (Levendusky)

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
        name="svpls",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


part = party_sorting


def cheatsheet() -> str:
    return "party_sorting({}) -> Party sorting index (Levendusky)"
