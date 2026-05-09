"""Spatial von Mises distribution"""

import numpy as np

from ._containers import DescriptiveResult


def von_mises_sp(data, *, method="default"):
    """Spatial von Mises distribution

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
        name="zxvms",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


von_ = von_mises_sp


def cheatsheet() -> str:
    return "von_mises_sp({}) -> Spatial von Mises distribution"
