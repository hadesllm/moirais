"""Two-issue salience model"""

import numpy as np

from ._containers import DescriptiveResult


def salience_2issue(data, *, method="default"):
    """Two-issue salience model

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
        name="svsl2",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


sali = salience_2issue


def cheatsheet() -> str:
    return "salience_2issue({}) -> Two-issue salience model"
