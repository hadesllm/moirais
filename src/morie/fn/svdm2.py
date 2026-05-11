"""Second dimensionality test"""

import numpy as np

from ._containers import DescriptiveResult


def dim_test_2(data, *, method="default"):
    """Second dimensionality test

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
        name="svdm2",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


dim_ = dim_test_2


def cheatsheet() -> str:
    return "dim_test_2({}) -> Second dimensionality test"
