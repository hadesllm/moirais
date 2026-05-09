"""Spatial circular mean"""

import numpy as np

from ._containers import DescriptiveResult


def circular_mean_sp(data, *, method="default"):
    """Spatial circular mean

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
        name="zxcmn",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


circ = circular_mean_sp


def cheatsheet() -> str:
    return "circular_mean_sp({}) -> Spatial circular mean"
