# moirais.fn — function file (hadesllm/moirais)
"""Double centering matrix B"""

import numpy as np

from ._containers import DescriptiveResult


def double_center(data, *, method="default"):
    """Double centering matrix B

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
        name="msdbc",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


doub = double_center


def cheatsheet() -> str:
    return "double_center({}) -> Double centering matrix B"
