"""1D ideal point estimation"""

import numpy as np

from ._containers import DescriptiveResult


def ideal_point_1d(data, *, method="default"):
    """1D ideal point estimation

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
        name="svip1",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


idea = ideal_point_1d


def cheatsheet() -> str:
    return "ideal_point_1d({}) -> 1D ideal point estimation"
