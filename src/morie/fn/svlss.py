"""Spatial loss function (quadratic/city-block)"""

import numpy as np

from ._containers import DescriptiveResult


def loss_function(data, *, method="default"):
    """Spatial loss function (quadratic/city-block)

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
        name="svlss",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


loss = loss_function


def cheatsheet() -> str:
    return "loss_function({}) -> Spatial loss function (quadratic/city-block)"
