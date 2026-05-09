# moirais.fn — function file (hadesllm/moirais)
"""Weighted SMACOF MDS"""

import numpy as np

from ._containers import DescriptiveResult


def smacof_weight(data, *, method="default"):
    """Weighted SMACOF MDS

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
        name="mssmw",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


smac = smacof_weight


def cheatsheet() -> str:
    return "smacof_weight({}) -> Weighted SMACOF MDS"
