"""Banks set computation"""

import numpy as np

from ._containers import DescriptiveResult


def banks_set(data, *, method="default"):
    """Banks set computation

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
        name="svbnk",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


bank = banks_set


def cheatsheet() -> str:
    return "banks_set({}) -> Banks set computation"
