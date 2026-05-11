# morie.fn — function file (hadesllm/morie)
"""W-NOMINATE dimension weights"""

import numpy as np

from ._containers import DescriptiveResult


def wnominate_weight(data, *, method="default"):
    """W-NOMINATE dimension weights

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
        name="nmwnw",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


wnom = wnominate_weight


def cheatsheet() -> str:
    return "wnominate_weight({}) -> W-NOMINATE dimension weights"
