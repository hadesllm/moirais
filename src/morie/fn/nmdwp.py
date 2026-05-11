# morie.fn — function file (hadesllm/morie)
"""DW-NOMINATE polarization"""

import numpy as np

from ._containers import DescriptiveResult


def dwnominate_polar(data, *, method="default"):
    """DW-NOMINATE polarization

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
        name="nmdwp",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


dwno = dwnominate_polar


def cheatsheet() -> str:
    return "dwnominate_polar({}) -> DW-NOMINATE polarization"
