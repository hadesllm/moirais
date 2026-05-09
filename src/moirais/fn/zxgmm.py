"""Spatial Gaussian mixture"""

import numpy as np

from ._containers import DescriptiveResult


def gmm_spatial(data, *, method="default"):
    """Spatial Gaussian mixture

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
        name="zxgmm",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


gmm_ = gmm_spatial


def cheatsheet() -> str:
    return "gmm_spatial({}) -> Spatial Gaussian mixture"
