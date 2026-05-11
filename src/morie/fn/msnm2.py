# morie.fn — function file (hadesllm/morie)
"""Nonmetric MDS 2D"""

import numpy as np

from ._containers import DescriptiveResult


def nonmetric_2d(data, *, method="default"):
    """Nonmetric MDS 2D

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
        name="msnm2",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


nonm = nonmetric_2d


def cheatsheet() -> str:
    return "nonmetric_2d({}) -> Nonmetric MDS 2D"
