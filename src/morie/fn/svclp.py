"""Cutting plane in 3D"""

import numpy as np

from ._containers import DescriptiveResult


def cut_plane(data, *, method="default"):
    """Cutting plane in 3D

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
        name="svclp",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


cut_ = cut_plane


def cheatsheet() -> str:
    return "cut_plane({}) -> Cutting plane in 3D"
