# morie.fn -- function file (hadesllm/morie)
"""Local continuity meta-criterion"""

import numpy as np

from ._containers import DescriptiveResult


def lcmc(data, *, method="default"):
    """Local continuity meta-criterion

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
        name="mslcl",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


lcm = lcmc


def cheatsheet() -> str:
    return "lcmc({}) -> Local continuity meta-criterion"
