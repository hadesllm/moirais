"""Rubinstein spatial bargaining"""

import numpy as np

from ._containers import DescriptiveResult


def rubinstein_sp(data, *, method="default"):
    """Rubinstein spatial bargaining

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
        name="svrub",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


rubi = rubinstein_sp


def cheatsheet() -> str:
    return "rubinstein_sp({}) -> Rubinstein spatial bargaining"
