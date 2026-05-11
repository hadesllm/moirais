"""Committee median voter model"""

import numpy as np

from ._containers import DescriptiveResult


def committee_med(data, *, method="default"):
    """Committee median voter model

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
        name="svcmp",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


comm = committee_med


def cheatsheet() -> str:
    return "committee_med({}) -> Committee median voter model"
