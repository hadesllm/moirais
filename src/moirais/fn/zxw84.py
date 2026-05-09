"""WGS84 to local tangent plane"""

import numpy as np

from ._containers import DescriptiveResult


def wgs84_to_local(data, *, method="default"):
    """WGS84 to local tangent plane

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
        name="zxw84",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


wgs8 = wgs84_to_local


def cheatsheet() -> str:
    return "wgs84_to_local({}) -> WGS84 to local tangent plane"
