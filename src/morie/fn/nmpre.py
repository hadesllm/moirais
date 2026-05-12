# morie.fn -- function file (hadesllm/morie)
"""Proportional Reduction in Error"""

import numpy as np

from ._containers import DescriptiveResult


def pre_stat(data, *, method="default"):
    """Proportional Reduction in Error

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
        name="nmpre",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


pre_ = pre_stat


def cheatsheet() -> str:
    return "pre_stat({}) -> Proportional Reduction in Error"
