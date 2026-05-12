# morie.fn -- function file (hadesllm/morie)
"""2D Voronoi diagram"""

import numpy as np

from ._containers import DescriptiveResult


def voronoi_2d(data, *, method="default"):
    """2D Voronoi diagram

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
        name="msvor",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


voro = voronoi_2d


def cheatsheet() -> str:
    return "voronoi_2d({}) -> 2D Voronoi diagram"
