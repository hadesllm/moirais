"""Functional data analysis spatial"""

import numpy as np

from ._containers import DescriptiveResult


def fda_spatial(data, *, method="default"):
    """Functional data analysis spatial

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
        name="zxfda",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


fda_ = fda_spatial


def cheatsheet() -> str:
    return "fda_spatial({}) -> Functional data analysis spatial"
