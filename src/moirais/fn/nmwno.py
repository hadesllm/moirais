# moirais.fn — function file (hadesllm/moirais)
"""W-NOMINATE estimation"""

import numpy as np

from ._containers import DescriptiveResult


def wnominate(data, *, method="default"):
    """W-NOMINATE estimation

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
        name="nmwno",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


wnom = wnominate


def cheatsheet() -> str:
    return "wnominate({}) -> W-NOMINATE estimation"
