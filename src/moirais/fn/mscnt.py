# moirais.fn — function file (hadesllm/moirais)
"""Continuity metric"""

import numpy as np

from ._containers import DescriptiveResult


def continuity(data, *, method="default"):
    """Continuity metric

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
        name="mscnt",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


cont = continuity


def cheatsheet() -> str:
    return "continuity({}) -> Continuity metric"
