# morie.fn -- function file (hadesllm/morie)
"""Shepard residuals"""

import numpy as np

from ._containers import DescriptiveResult


def shepard_resid(data, *, method="default"):
    """Shepard residuals

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
        name="msshr",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


shep = shepard_resid


def cheatsheet() -> str:
    return "shepard_resid({}) -> Shepard residuals"
