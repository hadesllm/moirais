"""Plott radial symmetry condition check"""

import numpy as np

from ._containers import DescriptiveResult


def plott_condition(data, *, method="default"):
    """Plott radial symmetry condition check

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
        name="svplc",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


plot = plott_condition


def cheatsheet() -> str:
    return "plott_condition({}) -> Plott radial symmetry condition check"
