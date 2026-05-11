"""Ranked probability score spatial"""

import numpy as np

from ._containers import DescriptiveResult


def rank_prob_score(data, *, method="default"):
    """Ranked probability score spatial

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
        name="svrps",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


rank = rank_prob_score


def cheatsheet() -> str:
    return "rank_prob_score({}) -> Ranked probability score spatial"
