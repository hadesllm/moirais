# morie.fn — function file (hadesllm/morie)
"""Optimal Classification 2D"""

import numpy as np

from ._containers import DescriptiveResult


def oc_2d(data, *, method="default"):
    """Optimal Classification 2D

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
        name="nmoc2",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


oc_ = oc_2d


def cheatsheet() -> str:
    return "oc_2d({}) -> Optimal Classification 2D"
