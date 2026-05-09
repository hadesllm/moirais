# moirais.fn — function file (hadesllm/moirais)
"""Three-way INDSCAL"""

import numpy as np

from ._containers import DescriptiveResult


def indscal_3way(data, *, method="default"):
    """Three-way INDSCAL

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
        name="msink",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


inds = indscal_3way


def cheatsheet() -> str:
    return "indscal_3way({}) -> Three-way INDSCAL"
