"""Top cycle set computation"""

import numpy as np

from ._containers import DescriptiveResult


def top_cycle_set(data, *, method="default"):
    """Top cycle set computation

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
        name="svtcs",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


top_ = top_cycle_set


def cheatsheet() -> str:
    return "top_cycle_set({}) -> Top cycle set computation"
