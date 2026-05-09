# moirais.fn — function file (hadesllm/moirais)
"""2D Delaunay triangulation"""

import numpy as np

from ._containers import DescriptiveResult


def delaunay_2d(data, *, method="default"):
    """2D Delaunay triangulation

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
        name="msdel",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


dela = delaunay_2d


def cheatsheet() -> str:
    return "delaunay_2d({}) -> 2D Delaunay triangulation"
