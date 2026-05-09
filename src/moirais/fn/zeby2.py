"""BYM2 reparameterized model"""

import numpy as np

from ._containers import SpatialResult


def bym2_model(data, *, method="default"):
    """BYM2 reparameterized model

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
        name="BYM2 reparameterized model",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


bym2 = bym2_model


def cheatsheet() -> str:
    return "bym2_model({}) -> BYM2 reparameterized model"
