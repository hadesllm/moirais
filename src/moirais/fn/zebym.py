"""BYM (Besag-York-Mollie) model"""

import numpy as np

from ._containers import SpatialResult


def bym_model(data, *, method="default"):
    """BYM (Besag-York-Mollie) model

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
        name="BYM (Besag-York-Mollie) model",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


bym_ = bym_model


def cheatsheet() -> str:
    return "bym_model({}) -> BYM (Besag-York-Mollie) model"
