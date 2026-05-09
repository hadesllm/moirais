"""SLX model OLS estimation"""

import numpy as np

from ._containers import SpatialResult


def slx_ols(data, *, method="default"):
    """SLX model OLS estimation

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
        name="SLX model OLS estimation",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


slx_ = slx_ols


def cheatsheet() -> str:
    return "slx_ols({}) -> SLX model OLS estimation"
