"""Wind rose directional stats"""

import numpy as np

from ._containers import DescriptiveResult


def wind_rose(data, *, method="default"):
    """Wind rose directional stats

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
        name="zxwnd",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


wind = wind_rose


def cheatsheet() -> str:
    return "wind_rose({}) -> Wind rose directional stats"
