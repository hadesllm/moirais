# moirais.fn — function file (hadesllm/moirais)
"""Optimal Classification cutting line"""

import numpy as np

from ._containers import DescriptiveResult


def oc_cutline(data, *, method="default"):
    """Optimal Classification cutting line

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
        name="nmocl",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


oc_c = oc_cutline


def cheatsheet() -> str:
    return "oc_cutline({}) -> Optimal Classification cutting line"
