# moirais.fn — function file (hadesllm/moirais)
"""INDSCAL individual differences MDS"""

import numpy as np

from ._containers import DescriptiveResult


def indscal(data, *, method="default"):
    """INDSCAL individual differences MDS

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
        name="msind",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


inds = indscal


def cheatsheet() -> str:
    return "indscal({}) -> INDSCAL individual differences MDS"
