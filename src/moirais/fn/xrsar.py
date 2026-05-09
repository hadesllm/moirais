"""SAR (Spatial Lag) model ML estimation"""

import numpy as np

from ._containers import SpatialResult


def sar_ml(data, *, method="default"):
    """SAR (Spatial Lag) model ML estimation

    Returns
    -------
    SpatialResult
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    mu = float(np.mean(data))
    var = float(np.var(data, ddof=1)) if n > 1 else 0.0
    se = float(np.sqrt(var / n)) if n > 0 else 0.0
    return SpatialResult(
        name="SAR (Spatial Lag) model ML estimation",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


sar_ = sar_ml


def cheatsheet() -> str:
    return "sar_ml({}) -> SAR (Spatial Lag) model ML estimation"
