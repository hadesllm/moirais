"""Minimum winning coalition size"""

import numpy as np

from ._containers import DescriptiveResult


def coalition_size(data, *, method="default"):
    """Minimum winning coalition size

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
        name="svcls",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


coal = coalition_size


def cheatsheet() -> str:
    return "coalition_size({}) -> Minimum winning coalition size"
